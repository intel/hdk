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
#include <armadillo>

namespace costmodel {

struct LinearRegression::PrivateImpl {
  arma::mat buildFeaturesMatrix(const std::vector<Detail::Measurement>& measurement);
  arma::vec buildTargets(const std::vector<Detail::Measurement>& measurement);

  arma::vec weights;
};

LinearRegression::LinearRegression(const std::vector<Detail::Measurement>& measurement)
    : ExtrapolationModel(measurement), pimpl_(new PrivateImpl()) {
  buildRegressionCoefficients();
}

LinearRegression::LinearRegression(std::vector<Detail::Measurement>&& measurement)
    : ExtrapolationModel(std::move(measurement)), pimpl_(new PrivateImpl()) {
  buildRegressionCoefficients();
}

LinearRegression::~LinearRegression() = default;

size_t LinearRegression::getExtrapolatedData(size_t bytes) {
  arma::vec x = {1.0, static_cast<double>(bytes)};
  return static_cast<size_t>(arma::dot(x, pimpl_->weights));
}

void LinearRegression::buildRegressionCoefficients() {
  // y(x) = w0 + \sum xj * wj = x^T * w
  arma::mat X = pimpl_->buildFeaturesMatrix(measurement_);
  arma::vec y = pimpl_->buildTargets(measurement_);

  pimpl_->weights = arma::inv(X.t() * X) * X.t() * y;
}

arma::mat LinearRegression::PrivateImpl::buildFeaturesMatrix(
    const std::vector<Detail::Measurement>& measurement) {
  // x = (1, bytes)
  const size_t featuresSize = 2;
  arma::mat X(measurement.size(), featuresSize, arma::fill::ones);

  for (size_t row = 0; row < measurement.size(); row++) {
    X(row, 1) = static_cast<double>(measurement[row].bytes);
  }

  return X;
}

arma::vec LinearRegression::PrivateImpl::buildTargets(
    const std::vector<Detail::Measurement>& measurement) {
  arma::vec y(measurement.size());
  for (size_t m = 0; m < measurement.size(); m++) {
    y(m) = measurement[m].milliseconds;
  }

  return y;
}

}  // namespace costmodel
