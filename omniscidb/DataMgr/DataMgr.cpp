/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    DataMgr.cpp
 * @author Todd Mostak <todd@mapd.com>
 */

#include "DataMgr/DataMgr.h"
#include "BufferMgr/CpuBufferMgr/CpuBufferMgr.h"
#include "BufferMgr/CpuBufferMgr/TieredCpuBufferMgr.h"
#include "BufferMgr/GpuBufferMgr/GpuBufferMgr.h"
#include "CudaMgr/CudaMgr.h"
#include "PersistentStorageMgr/PersistentStorageMgr.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include <boost/filesystem.hpp>

#include <algorithm>
#include <limits>
#include <numeric>

namespace Data_Namespace {

DataMgr::DataMgr(const Config& config, const size_t numReaderThreads)
    : current_device_mgr_(nullptr)
    , has_gpus_(false)
    , reservedGpuMem_(config.mem.gpu.reserved_mem_bytes)
    , buffer_provider_(std::make_unique<DataMgrBufferProvider>(this))
    , data_provider_(std::make_unique<DataMgrDataProvider>(this)) {
  populateDeviceMgrs(config);
  populateMgrs(config, numReaderThreads);
  createTopLevelMetadata();
}

DataMgr::~DataMgr() {
  for (auto& [p, ctx] : device_contexts_) {
    for (size_t device = 0; device < ctx->buffer_mgrs.size(); device++) {
      delete ctx->buffer_mgrs[device];
    }
  }

  int num_levels =
      bufferMgrs_.size() == GPU_LEVEL + 1 ? CPU_LEVEL + 1 : bufferMgrs_.size();
  for (int level = num_levels - 1; level >= 0; --level) {
    for (size_t device = 0; device < bufferMgrs_[level].size(); device++) {
      delete bufferMgrs_[level][device];
    }
  }
}

DataMgr::SystemMemoryUsage DataMgr::getSystemMemoryUsage() const {
  SystemMemoryUsage usage;

#ifdef __linux__

  // Determine Linux available memory and total memory.
  // Available memory is different from free memory because
  // when Linux sees free memory, it tries to use it for
  // stuff like disk caching. However, the memory is not
  // reserved and is still available to be allocated by
  // user processes.
  // Parsing /proc/meminfo for this info isn't very elegant
  // but as a virtual file it should be reasonably fast.
  // See also:
  //   https://github.com/torvalds/linux/commit/34e431b0ae398fc54ea69ff85ec700722c9da773
  ProcMeminfoParser mi;
  usage.free = mi["MemAvailable"];
  usage.total = mi["MemTotal"];

  // Determine process memory in use.
  // See also:
  //   https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
  //   http://man7.org/linux/man-pages/man5/proc.5.html
  int64_t size = 0;
  int64_t resident = 0;
  int64_t shared = 0;

  std::ifstream fstatm("/proc/self/statm");
  fstatm >> size >> resident >> shared;
  fstatm.close();

  long page_size =
      sysconf(_SC_PAGE_SIZE);  // in case x86-64 is configured to use 2MB pages

  usage.resident = resident * page_size;
  usage.vtotal = size * page_size;
  usage.regular = (resident - shared) * page_size;
  usage.shared = shared * page_size;

  ProcBuddyinfoParser bi;
  usage.frag = bi.getFragmentationPercent();

#else

  usage.total = 0;
  usage.free = 0;
  usage.resident = 0;
  usage.vtotal = 0;
  usage.regular = 0;
  usage.shared = 0;
  usage.frag = 0;

#endif

  return usage;
}

size_t DataMgr::getTotalSystemMemory() {
#ifdef __APPLE__
  int mib[2];
  size_t physical_memory;
  size_t length;
  // Get the Physical memory size
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  length = sizeof(size_t);
  sysctl(mib, 2, &physical_memory, &length, NULL, 0);
  return physical_memory;
#elif defined(_MSC_VER)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
#else  // Linux
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
#endif
}

void DataMgr::allocateCpuBufferMgr(int32_t device_id,
                                   bool enable_tiered_cpu_mem,
                                   size_t total_cpu_size,
                                   size_t minCpuSlabSize,
                                   size_t maxCpuSlabSize,
                                   size_t page_size,
                                   const CpuTierSizeVector& cpu_tier_sizes) {
  GpuMgr* gpuMgr = getGpuMgr();

  if (enable_tiered_cpu_mem) {
    bufferMgrs_[MemoryLevel::CPU_LEVEL].push_back(
        new Buffer_Namespace::TieredCpuBufferMgr(
            0,
            total_cpu_size,
            gpuMgr,
            minCpuSlabSize,
            maxCpuSlabSize,
            page_size,
            cpu_tier_sizes,
            bufferMgrs_[MemoryLevel::DISK_LEVEL][0]));
  } else {
    bufferMgrs_[MemoryLevel::CPU_LEVEL].push_back(
        new Buffer_Namespace::CpuBufferMgr(0,
                                           total_cpu_size,
                                           gpuMgr,
                                           minCpuSlabSize,
                                           maxCpuSlabSize,
                                           page_size,
                                           bufferMgrs_[MemoryLevel::DISK_LEVEL][0]));
  }
}

void DataMgr::populateDeviceMgrs(const Config& config) {
  if (config.exec.cpu_only)
    return;

#ifdef HAVE_CUDA
  try {
    device_mgrs_[GpuMgrPlatform::CUDA] =
        std::make_unique<CudaMgr_Namespace::CudaMgr>(-1, 0);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to initialize CUDA GPU: " << e.what();
    device_mgrs_.erase(GpuMgrPlatform::CUDA);
  }
#endif
#ifdef HAVE_L0
  try {
    device_mgrs_[GpuMgrPlatform::L0] = std::make_unique<l0::L0Manager>();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to initialize L0 GPU: " << e.what();
    device_mgrs_.erase(GpuMgrPlatform::L0);
  }
#endif

  has_gpus_ = device_mgrs_.size();
  if (!has_gpus_) {
    LOG(INFO) << "None of the passed GpuMgr instances is valid, falling back to "
                 "CPU-only mode.";
    return;
  }
  current_device_mgr_ = device_mgrs_.begin()->second.get();
}

void DataMgr::populateMgrs(const Config& config,
                           const size_t userSpecifiedNumReaderThreads) {
  // no need for locking, as this is only called in the constructor
  bufferMgrs_.resize(2);
  levelSizes_.resize(2);
  bufferMgrs_[MemoryLevel::DISK_LEVEL].push_back(
      new PersistentStorageMgr(userSpecifiedNumReaderThreads));

  levelSizes_[DISK_LEVEL] = 1;
  size_t page_size{512};
  size_t cpuBufferSize = config.mem.cpu.max_size;
  if (cpuBufferSize == 0) {  // if size is not specified
    const auto total_system_memory = getTotalSystemMemory();
    VLOG(1) << "Detected " << (float)total_system_memory / (1024 * 1024)
            << "M of total system memory.";
    cpuBufferSize = total_system_memory *
                    0.8;  // should get free memory instead of this ugly heuristic
  }
  size_t minCpuSlabSize = std::min(config.mem.cpu.min_slab_size, cpuBufferSize);
  minCpuSlabSize = (minCpuSlabSize / page_size) * page_size;
  size_t maxCpuSlabSize = std::min(config.mem.cpu.max_slab_size, cpuBufferSize);
  maxCpuSlabSize = (maxCpuSlabSize / page_size) * page_size;
  LOG(INFO) << "Min CPU Slab Size is " << (float)minCpuSlabSize / (1024 * 1024) << "MB";
  LOG(INFO) << "Max CPU Slab Size is " << (float)maxCpuSlabSize / (1024 * 1024) << "MB";
  LOG(INFO) << "Max memory pool size for CPU is " << (float)cpuBufferSize / (1024 * 1024)
            << "MB";

  CpuTierSizeVector cpu_tier_sizes(numCpuTiers, 0);
  cpu_tier_sizes[CpuTier::DRAM] = cpuBufferSize;

  if (config.mem.cpu.enable_tiered_cpu_mem) {
    cpu_tier_sizes[CpuTier::PMEM] = config.mem.cpu.pmem_size;
    LOG(INFO) << "Max memory pool size for PMEM is "
              << (float)config.mem.cpu.pmem_size / (1024 * 1024) << "MB";
  }

  auto total_cpu_size = std::reduce(cpu_tier_sizes.begin(), cpu_tier_sizes.end());
  levelSizes_[CPU_LEVEL] = 1;

  if (has_gpus_) {
    LOG(INFO) << "Reserved GPU memory is " << (float)reservedGpuMem_ / (1024 * 1024)
              << "MB includes render buffer allocation";
    bufferMgrs_.resize(3);
    levelSizes_.resize(3);
    allocateCpuBufferMgr(0,
                         config.mem.cpu.enable_tiered_cpu_mem,
                         total_cpu_size,
                         minCpuSlabSize,
                         maxCpuSlabSize,
                         page_size,
                         cpu_tier_sizes);

    for (auto& [p, mgr] : device_mgrs_) {
      LOG(DEBUG2) << "Creating device context for platform "
                  << (p == GpuMgrPlatform::L0 ? "L0" : "CUDA");
      device_contexts_[p] = std::make_unique<GpuMgrContext>();
      auto& device_context = device_contexts_[p];
      device_context->gpu_mgr = mgr.get();
      int num_gpus = mgr->getDeviceCount();
      device_context->gpu_count = num_gpus;
      for (int gpu_num = 0; gpu_num < num_gpus; ++gpu_num) {
        size_t device_mem_size = 0;
        // TODO: get rid of manager-specific branches by introducing some kind of device
        // properties in GpuMgr
        switch (p) {
          case GpuMgrPlatform::CUDA:
            device_mem_size = getCudaMgr()->getDeviceProperties(gpu_num)->globalMem;
            break;
          case GpuMgrPlatform::L0:
            device_mem_size = getL0Mgr()->getMaxAllocationSize(gpu_num);
            page_size = getL0Mgr()->getPageSize(gpu_num);
            break;
          default:
            CHECK(false);
        }

        size_t gpu_max_mem_size = config.mem.gpu.max_size;
        if (gpu_max_mem_size == 0) {
          CHECK_GT(device_mem_size, reservedGpuMem_);
          gpu_max_mem_size = device_mem_size - reservedGpuMem_;
        }

        size_t minGpuSlabSize = std::min(config.mem.gpu.min_slab_size, gpu_max_mem_size);
        minGpuSlabSize = (minGpuSlabSize / page_size) * page_size;
        size_t maxGpuSlabSize = std::min(config.mem.gpu.max_slab_size, gpu_max_mem_size);
        maxGpuSlabSize = (maxGpuSlabSize / page_size) * page_size;
        LOG(INFO) << "Min GPU Slab size for GPU " << gpu_num << " is "
                  << (float)minGpuSlabSize / (1024 * 1024) << "MB";
        LOG(INFO) << "Max GPU Slab size for GPU " << gpu_num << " is "
                  << (float)maxGpuSlabSize / (1024 * 1024) << "MB";
        LOG(INFO) << "Max memory pool size for GPU " << gpu_num << " is "
                  << (float)gpu_max_mem_size / (1024 * 1024) << "MB";

        device_context->buffer_mgrs.push_back(
            new Buffer_Namespace::GpuBufferMgr(gpu_num,
                                               gpu_max_mem_size,
                                               mgr.get(),
                                               minGpuSlabSize,
                                               maxGpuSlabSize,
                                               page_size,
                                               bufferMgrs_[MemoryLevel::CPU_LEVEL][0]));
      }
    }
    setGpuMgrContext(device_mgrs_.begin()->second->getPlatform());
  } else {
    allocateCpuBufferMgr(0,
                         config.mem.cpu.enable_tiered_cpu_mem,
                         total_cpu_size,
                         minCpuSlabSize,
                         maxCpuSlabSize,
                         page_size,
                         cpu_tier_sizes);
  }
}

void DataMgr::convertDB(const std::string basePath) {
  UNREACHABLE();
}

void DataMgr::createTopLevelMetadata() const {}

std::vector<Buffer_Namespace::MemoryInfo> DataMgr::getMemoryInfo(
    const MemoryLevel memLevel) {
  std::vector<Buffer_Namespace::MemoryInfo> mem_info;
  if (memLevel == MemoryLevel::CPU_LEVEL) {
    Buffer_Namespace::CpuBufferMgr* cpu_buffer =
        dynamic_cast<Buffer_Namespace::CpuBufferMgr*>(
            bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
    CHECK(cpu_buffer);
    mem_info.push_back(cpu_buffer->getMemoryInfo());
  } else if (has_gpus_) {
    int num_gpus = getGpuMgr()->getDeviceCount();
    CHECK_EQ(num_gpus, bufferMgrs_[MemoryLevel::GPU_LEVEL].size());
    for (int gpu_num = 0; gpu_num < num_gpus; ++gpu_num) {
      Buffer_Namespace::BufferMgr* gpu_buffer =
          dynamic_cast<Buffer_Namespace::BufferMgr*>(
              bufferMgrs_[MemoryLevel::GPU_LEVEL][gpu_num]);
      CHECK(gpu_buffer);
      mem_info.push_back(gpu_buffer->getMemoryInfo());
    }
  }
  return mem_info;
}

std::string DataMgr::dumpLevel(const MemoryLevel memLevel) {
  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    int num_gpus = getGpuMgr()->getDeviceCount();
    std::ostringstream tss;
    for (int gpu_num = 0; gpu_num < num_gpus; ++gpu_num) {
      tss << bufferMgrs_[memLevel][gpu_num]->printSlabs();
    }
    return tss.str();
  } else {
    return bufferMgrs_[memLevel][0]->printSlabs();
  }
}

void DataMgr::clearMemory(const MemoryLevel memLevel) {
  // if gpu we need to iterate through all the buffermanagers for each card
  if (memLevel == MemoryLevel::GPU_LEVEL) {
    if (has_gpus_) {
      int num_gpus = getGpuMgr()->getDeviceCount();
      for (int gpu_num = 0; gpu_num < num_gpus; ++gpu_num) {
        LOG(INFO) << "clear slabs on gpu " << gpu_num;
        auto buffer_mgr_for_gpu =
            dynamic_cast<Buffer_Namespace::BufferMgr*>(bufferMgrs_[memLevel][gpu_num]);
        CHECK(buffer_mgr_for_gpu);
        buffer_mgr_for_gpu->clearSlabs();
      }
    } else {
      LOG(WARNING) << "Unable to clear GPU memory: No GPUs detected";
    }
  } else {
    auto buffer_mgr_for_cpu =
        dynamic_cast<Buffer_Namespace::BufferMgr*>(bufferMgrs_[memLevel][0]);
    CHECK(buffer_mgr_for_cpu);
    buffer_mgr_for_cpu->clearSlabs();
  }
}

bool DataMgr::isBufferOnDevice(const ChunkKey& key,
                               const MemoryLevel memLevel,
                               const int deviceId) {
  return bufferMgrs_[memLevel][deviceId]->isBufferOnDevice(key);
}

GpuMgr* DataMgr::getGpuMgr(GpuMgrPlatform name) const {
  GpuMgr* res = nullptr;
  if (device_mgrs_.count(name)) {
    res = device_mgrs_.at(name).get();
    CHECK_EQ(res->getPlatform(), name) << "Mapping of GPU managers names is incorrect";
  }

  return res;
}

void DataMgr::setGpuMgrContext(GpuMgrPlatform name) {
  CHECK(device_contexts_.count(name));
  CHECK(device_mgrs_.count(name));
  current_device_mgr_ = device_mgrs_.at(name).get();
  bufferMgrs_[MemoryLevel::GPU_LEVEL] = device_contexts_.at(name)->buffer_mgrs;
  levelSizes_[MemoryLevel::GPU_LEVEL] = device_contexts_.at(name)->gpu_count;
  LOG(INFO) << "Set GPU manager context to " << name;
}

void DataMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                              const ChunkKey& keyPrefix) {
  bufferMgrs_[MemoryLevel::DISK_LEVEL][0]->getChunkMetadataVecForKeyPrefix(
      chunkMetadataVec, keyPrefix);
}

AbstractBuffer* DataMgr::createChunkBuffer(const ChunkKey& key,
                                           const MemoryLevel memoryLevel,
                                           const int deviceId,
                                           const size_t page_size) {
  int level = static_cast<int>(memoryLevel);
  return bufferMgrs_[level][deviceId]->createBuffer(key, page_size);
}

AbstractBuffer* DataMgr::getChunkBuffer(const ChunkKey& key,
                                        const MemoryLevel memoryLevel,
                                        const int deviceId,
                                        const size_t numBytes) {
  const auto level = static_cast<size_t>(memoryLevel);
  CHECK_LT(level, levelSizes_.size());     // make sure we have a legit buffermgr
  CHECK_LT(deviceId, levelSizes_[level]);  // make sure we have a legit buffermgr
  return bufferMgrs_[level][deviceId]->getBuffer(key, numBytes);
}

void DataMgr::deleteChunksWithPrefix(const ChunkKey& keyPrefix) {
  int numLevels = bufferMgrs_.size();
  for (int level = numLevels - 1; level >= 0; --level) {
    for (int device = 0; device < levelSizes_[level]; ++device) {
      bufferMgrs_[level][device]->deleteBuffersWithPrefix(keyPrefix);
    }
  }
}

// only deletes the chunks at the given memory level
void DataMgr::deleteChunksWithPrefix(const ChunkKey& keyPrefix,
                                     const MemoryLevel memLevel) {
  if (bufferMgrs_.size() <= memLevel) {
    return;
  }
  for (int device = 0; device < levelSizes_[memLevel]; ++device) {
    bufferMgrs_[memLevel][device]->deleteBuffersWithPrefix(keyPrefix);
  }
}

AbstractBuffer* DataMgr::alloc(const MemoryLevel memoryLevel,
                               const int deviceId,
                               const size_t numBytes) {
  const auto level = static_cast<int>(memoryLevel);
  CHECK_LT(deviceId, levelSizes_[level]);
  return bufferMgrs_[level][deviceId]->alloc(numBytes);
}

void DataMgr::free(AbstractBuffer* buffer) {
  int level = static_cast<int>(buffer->getType());
  bufferMgrs_[level][buffer->getDeviceId()]->free(buffer);
}

void DataMgr::copy(AbstractBuffer* destBuffer, AbstractBuffer* srcBuffer) {
  destBuffer->write(srcBuffer->getMemoryPtr(),
                    srcBuffer->size(),
                    0,
                    srcBuffer->getType(),
                    srcBuffer->getDeviceId());
}

// could add function below to do arbitrary copies between buffers

// void DataMgr::copy(AbstractBuffer *destBuffer, const AbstractBuffer *srcBuffer, const
// size_t numBytes, const size_t destOffset, const size_t srcOffset) {
//} /

void DataMgr::setTableEpoch(const int db_id, const int tb_id, const int start_epoch) {
  UNREACHABLE();
}

size_t DataMgr::getTableEpoch(const int db_id, const int tb_id) {
  UNREACHABLE();
  return 0;
}

std::ostream& operator<<(std::ostream& os, const DataMgr::SystemMemoryUsage& mem_info) {
  os << "jsonlog ";
  os << "{";
  os << " \"name\": \"CPU Memory Info\",";
  os << " \"TotalMB\": " << mem_info.total / (1024. * 1024.) << ",";
  os << " \"FreeMB\": " << mem_info.free / (1024. * 1024.) << ",";
  os << " \"ProcessMB\": " << mem_info.resident / (1024. * 1024.) << ",";
  os << " \"VirtualMB\": " << mem_info.vtotal / (1024. * 1024.) << ",";
  os << " \"ProcessPlusSwapMB\": " << mem_info.regular / (1024. * 1024.) << ",";
  os << " \"ProcessSharedMB\": " << mem_info.shared / (1024. * 1024.) << ",";
  os << " \"FragmentationPercent\": " << mem_info.frag;
  os << " }";
  return os;
}

PersistentStorageMgr* DataMgr::getPersistentStorageMgr() const {
  return dynamic_cast<PersistentStorageMgr*>(bufferMgrs_[MemoryLevel::DISK_LEVEL][0]);
}

Buffer_Namespace::CpuBufferMgr* DataMgr::getCpuBufferMgr() const {
  return dynamic_cast<Buffer_Namespace::CpuBufferMgr*>(
      bufferMgrs_[MemoryLevel::CPU_LEVEL][0]);
}

const DictDescriptor* DataMgr::getDictMetadata(int dict_id, bool load_dict) const {
  return getPersistentStorageMgr()->getDictMetadata(dict_id, load_dict);
}

TableFragmentsInfo DataMgr::getTableMetadata(int db_id, int table_id) const {
  return getPersistentStorageMgr()->getTableMetadata(db_id, table_id);
}

}  // namespace Data_Namespace
