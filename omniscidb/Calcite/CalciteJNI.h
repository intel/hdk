/*
 * Copyright 2022 Intel Corporation.
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

#pragma once

#include <future>
#include <queue>

#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "SchemaMgr/SchemaProvider.h"
#include "Shared/Config.h"

struct FilterPushDownInfo {
  int input_prev;
  int input_start;
  int input_next;
};

class CalciteJNI {
 public:
  CalciteJNI(SchemaProviderPtr schema_provider,
             ConfigPtr config,
             const std::string& udf_filename = "",
             size_t calcite_max_mem_mb = 1024);
  ~CalciteJNI();

  std::string process(const std::string& db_name,
                      const std::string& sql_string,
                      const std::vector<FilterPushDownInfo>& filter_push_down_info = {},
                      const bool legacy_syntax = false,
                      const bool is_explain = false,
                      const bool is_view_optimize = false);

  std::string getExtensionFunctionWhitelist();
  std::string getUserDefinedFunctionWhitelist();
  std::string getRuntimeExtensionFunctionWhitelist();
  void setRuntimeExtensionFunctions(const std::vector<ExtensionFunction>& udfs,
                                    bool is_runtime = true);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Run CalciteJNI on a single thread.
 */
class CalciteWorker {
 public:
  using Task = std::packaged_task<std::string(CalciteJNI* calcite_jni)>;

  CalciteWorker(const CalciteWorker&) = delete;

  ~CalciteWorker() {
    {
      std::lock_guard<decltype(queue_mutex_)> lock(queue_mutex_);
      should_exit_ = true;
    }
    worker_cv_.notify_all();
    worker_.join();
  }

  // not thread safe. does this need a thread safety guarantee?
  static std::shared_ptr<CalciteWorker> initialize(SchemaProviderPtr schema_provider,
                                                   ConfigPtr config,
                                                   const std::string& udf_filename = "",
                                                   size_t calcite_max_mem_mb = 1024) {
    if (instance_) {
      throw std::runtime_error("Calcite worker thread is already initialized.");
    }
    instance_ = std::shared_ptr<CalciteWorker>(
        new CalciteWorker(schema_provider, config, udf_filename, calcite_max_mem_mb));
    return instance_;
  }

  static void teardown() { instance_ = nullptr; }

  std::string process(const std::string& db_name,
                      const std::string& sql_string,
                      const std::vector<FilterPushDownInfo>& filter_push_down_info = {},
                      const bool legacy_syntax = false,
                      const bool is_explain = false,
                      const bool is_view_optimize = false) {
    auto task = Task([&db_name,
                      &sql_string,
                      &filter_push_down_info,
                      legacy_syntax,
                      is_explain,
                      is_view_optimize](CalciteJNI* calcite_jni) {
      CHECK(calcite_jni);
      return calcite_jni->process(db_name,
                                  sql_string,
                                  filter_push_down_info,
                                  legacy_syntax,
                                  is_explain,
                                  is_view_optimize);
    });
    auto result = task.get_future();
    submitTaskToQueue(std::move(task));

    result.wait();
    return result.get();
  }

  std::string getExtensionFunctionWhitelist() {
    auto task = Task([](CalciteJNI* calcite_jni) {
      CHECK(calcite_jni);
      return calcite_jni->getExtensionFunctionWhitelist();
    });

    auto result = task.get_future();
    submitTaskToQueue(std::move(task));

    result.wait();
    return result.get();
  }

  std::string getUserDefinedFunctionWhitelist() {
    auto task = Task([](CalciteJNI* calcite_jni) {
      CHECK(calcite_jni);
      return calcite_jni->getUserDefinedFunctionWhitelist();
    });

    auto result = task.get_future();
    submitTaskToQueue(std::move(task));

    result.wait();
    return result.get();
  }

  std::string getRuntimeExtensionFunctionWhitelist() {
    auto task = Task([](CalciteJNI* calcite_jni) {
      CHECK(calcite_jni);
      return calcite_jni->getRuntimeExtensionFunctionWhitelist();
    });

    auto result = task.get_future();
    submitTaskToQueue(std::move(task));

    result.wait();
    return result.get();
  }

  void setRuntimeExtensionFunctions(const std::vector<ExtensionFunction>& udfs,
                                    bool is_runtime = true) {
    auto task = Task([&udfs, is_runtime](CalciteJNI* calcite_jni) {
      CHECK(calcite_jni);
      calcite_jni->setRuntimeExtensionFunctions(udfs, is_runtime);
      return "";  // all tasks return strings
    });
    submitTaskToQueue(std::move(task));
  }

 private:
  explicit CalciteWorker(SchemaProviderPtr schema_provider,
                         ConfigPtr config,
                         const std::string& udf_filename,
                         size_t calcite_max_mem_mb) {
    worker_ = std::thread(&CalciteWorker::worker,
                          this,
                          schema_provider,
                          config,
                          udf_filename,
                          calcite_max_mem_mb);
  }

  void worker(SchemaProviderPtr schema_provider,
              ConfigPtr config,
              const std::string& udf_filename,
              size_t calcite_max_mem_mb) {
    auto calcite_jni = std::make_unique<CalciteJNI>(
        schema_provider, config, udf_filename, calcite_max_mem_mb);

    std::unique_lock<std::mutex> lock(queue_mutex_);
    while (true) {
      worker_cv_.wait(lock, [this] { return !queue_.empty() || should_exit_; });

      if (should_exit_) {
        return;
      }

      if (!queue_.empty()) {
        auto task = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        task(calcite_jni.get());

        lock.lock();
      }
    }
  }

  void submitTaskToQueue(Task&& task) {
    std::unique_lock<decltype(queue_mutex_)> lock(queue_mutex_);

    queue_.push(std::move(task));

    lock.unlock();
    worker_cv_.notify_all();
  }

  std::mutex queue_mutex_;
  std::condition_variable worker_cv_;
  std::thread worker_;

  std::queue<Task> queue_;

  bool should_exit_{false};
  static std::shared_ptr<CalciteWorker> instance_;
};
