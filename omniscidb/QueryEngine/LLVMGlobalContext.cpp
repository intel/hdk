/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "LLVMGlobalContext.h"

#include <memory>
#include <mutex>

namespace {

llvm::orc::ThreadSafeContext g_global_context;
std::once_flag context_init_flag;

}  // namespace

llvm::orc::ThreadSafeContext& getGlobalLLVMThreadSafeContext() {
  std::call_once(context_init_flag, []() {
    auto global_context = std::make_unique<llvm::LLVMContext>();
    global_context->enableOpaquePointers();
    g_global_context = llvm::orc::ThreadSafeContext(std::move(global_context));
  });
  return g_global_context;
}
