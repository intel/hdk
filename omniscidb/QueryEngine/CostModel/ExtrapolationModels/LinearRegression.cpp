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

#include "LinearRegression.h"

namespace costmodel {

size_t LinearRegression::getExtrapolatedData(size_t bytes) {
  arma::vec x = {1.0, (double)bytes};
  return (size_t)arma::dot(x, w_);
}

void LinearRegression::buildRegressionCoefficients() {
  // y(x) = w0 + \sum xj * wj = x^T * w
  arma::mat X = buildFeaturesMatrix();
  arma::vec y = buildTargets();

  w_ = arma::inv(X.t() * X) * X.t() * y;
}

arma::mat LinearRegression::buildFeaturesMatrix() {
  // x = (1, bytes)
  const size_t featuresSize = 2;
  arma::mat X(measurement_.size(), featuresSize, arma::fill::ones);

  for (size_t row = 0; row < measurement_.size(); row++) {
    X(row, 1) = (double)measurement_[row].bytes;
  }

  return X;
}

arma::vec LinearRegression::buildTargets() {
  arma::vec y(measurement_.size());
  for (size_t m = 0; m < measurement_.size(); m++) {
    y(m) = measurement_[m].milliseconds;
  }

  return y;
}

}  // namespace costmodel
