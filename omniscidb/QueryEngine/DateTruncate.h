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

#ifndef QUERYENGINE_DATETRUNCATE_H
#define QUERYENGINE_DATETRUNCATE_H

#include <cstdint>

// DatetruncField must be synced with datetrunc_fname
enum DatetruncField {
  dtYEAR = 0,
  dtQUARTER,
  dtMONTH,
  dtDAY,
  dtHOUR,
  dtMINUTE,
  dtSECOND,
  dtMILLISECOND,
  dtMICROSECOND,
  dtNANOSECOND,
  dtMILLENNIUM,
  dtCENTURY,
  dtDECADE,
  dtWEEK,
  dtWEEK_SUNDAY,
  dtWEEK_SATURDAY,
  dtQUARTERDAY,
  dtINVALID
};

int64_t DateTruncate(DatetruncField field, const int64_t timeval);

// for usage in compiled and linked modules in the binary
int64_t truncate_high_precision_timestamp_to_date(const int64_t timeval,
                                                  const int64_t scale);

#endif  // QUERYENGINE_DATETRUNCATE_H
