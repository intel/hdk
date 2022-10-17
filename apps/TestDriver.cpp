/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "HDK.h"

#include <iostream>

int main(void) {
  std::cout << "Hello, world" << std::endl;

  auto hdk = HDK::init();
}
