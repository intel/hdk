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

#include "OSDependent/omnisci_fs.h"

#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "Logger/Logger.h"

namespace omnisci {

int get_page_size() {
  return getpagesize();
}

::FILE* popen(const char* command, const char* type) {
  return ::popen(command, type);
}

int32_t pclose(::FILE* fh) {
  return ::pclose(fh);
}

}  // namespace omnisci
