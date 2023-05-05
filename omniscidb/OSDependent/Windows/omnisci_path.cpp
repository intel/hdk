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

#include "OSDependent/omnisci_path.h"

#include <boost/filesystem/path.hpp>
#include <filesystem>

#include "Logger/Logger.h"

#include "Shared/clean_windows.h"

namespace omnisci {

std::string get_root_abs_path() {
  char abs_exe_path[MAX_PATH];
  auto path_len = GetModuleFileNameA(NULL, abs_exe_path, MAX_PATH);
  CHECK_GT(path_len, 0u);
  CHECK_LT(static_cast<size_t>(path_len), sizeof(abs_exe_path));
  boost::filesystem::path abs_exe_dir(std::string(abs_exe_path, path_len));
  abs_exe_dir.remove_filename();
  // When installed in conda env the path points python.exe that is located
  // in the root of conda environment.
  // When running tests in built repo the path points to omniscidb\Tests\{Debug|Release}\
  // which is three levels below "bin" directory with calcite jar file.
  // Because of this disbalance we try to look up QueryEngine/RuntimeFunctions.bc file.
  // QueryEngine is located either one level below "bin" or on the same level. When it is
  // below "bin" then this one more up level is done in CalciteJNI.
  const int UP_PATH_LEVELS = 2;
  for (int up_levels = 0; up_levels <= UP_PATH_LEVELS; up_levels++) {
    std::string target_path = abs_exe_dir.string() + "/QueryEngine/RuntimeFunctions.bc";
    if (std::filesystem::exists(target_path)) {
      break;
    }
    abs_exe_dir = abs_exe_dir.parent_path();
  }

  return abs_exe_dir.string();
}

}  // namespace omnisci
