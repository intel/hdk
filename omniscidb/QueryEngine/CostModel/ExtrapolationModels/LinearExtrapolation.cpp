/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "LinearExtrapolation.h"

#include <algorithm>

namespace costmodel {

size_t LinearExtrapolation::getExtrapolatedData(size_t bytes) {
  size_t id1, id2;
  Detail::Measurement tmp = {bytes, 0};

  auto iter = std::upper_bound(
      measurement_.begin(),
      measurement_.end(),
      tmp,
      [](const Detail::Measurement& lhs, const Detail::Measurement& rhs) {
        return lhs.bytes < rhs.bytes;
      });

  if (iter == measurement_.begin()) {
    id1 = 0;
    id2 = 1;
  } else if (iter == measurement_.end()) {
    id1 = measurement_.size() - 2;
    id2 = measurement_.size() - 1;
  } else {
    id2 = iter - measurement_.begin();
    id1 = id2 - 1;
  }

  size_t y1 = measurement_[id1].milliseconds, y2 = measurement_[id2].milliseconds;
  size_t x1 = measurement_[id1].bytes, x2 = measurement_[id2].bytes;

  return y1 + (static_cast<double>(bytes) - x1) / (x2 - x1) * (y2 - y1);
}

}  // namespace costmodel
