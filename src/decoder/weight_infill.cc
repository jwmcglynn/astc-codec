// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/decoder/weight_infill.h"

#include <array>
#include <cmath>
#include <utility>

#include "src/decoder/integer_sequence_codec.h"

namespace astc_codec {

namespace {

// The following functions are based on Section C.2.18 of the ASTC specification
unsigned int GetScaleFactorD(unsigned int block_dim) {
  return static_cast<unsigned int>(
      (1024.f + static_cast<float>(block_dim >> 1)) /
      static_cast<float>(block_dim - 1));
}

std::pair<unsigned int, unsigned int> GetGridSpaceCoordinates(
    Footprint footprint, unsigned int s, unsigned int t,
    unsigned int weight_dim_x, unsigned int weight_dim_y) {
  const int ds = GetScaleFactorD(footprint.Width());
  const int dt = GetScaleFactorD(footprint.Height());

  const int cs = ds * s;
  const int ct = dt * t;

  const int gs = (cs * (weight_dim_x - 1) + 32) >> 6;
  const int gt = (ct * (weight_dim_y - 1) + 32) >> 6;

  assert(gt < 1 << 8);
  assert(gs < 1 << 8);

  return std::make_pair(gs, gt);
}

// Returns the weight-grid values that are to be used for bilinearly
// interpolating the weight to its final value. If the returned value
// is equal to weight_dim_x * weight_dim_y, it may be ignored.
std::array<unsigned int, 4> BilerpGridPointsForWeight(
    const std::pair<unsigned int, unsigned int>& grid_space_coords,
    unsigned int weight_dim_x) {
  const unsigned int js = grid_space_coords.first >> 4;
  const unsigned int jt = grid_space_coords.second >> 4;

  std::array<unsigned int, 4> result;
  result[0] = js + weight_dim_x * jt;
  result[1] = js + weight_dim_x * jt + 1;
  result[2] = js + weight_dim_x * (jt + 1);
  result[3] = js + weight_dim_x * (jt + 1) + 1;

  return result;
}

std::array<unsigned int, 4> BilerpGridPointFactorsForWeight(
    const std::pair<unsigned int, unsigned int>& grid_space_coords) {
  const unsigned int fs = grid_space_coords.first & 0xF;
  const unsigned int ft = grid_space_coords.second & 0xF;

  std::array<unsigned int, 4> result;
  result[3] = (fs * ft + 8) >> 4;
  result[2] = ft - result[3];
  result[1] = fs - result[3];
  result[0] = 16 - fs - ft + result[3];

  assert(result[0] <= 16);
  assert(result[1] <= 16);
  assert(result[2] <= 16);
  assert(result[3] <= 16);

  return result;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////

unsigned int CountBitsForWeights(unsigned int weight_dim_x,
                                 unsigned int weight_dim_y,
                                 unsigned int target_weight_range) {
  const unsigned int num_weights = weight_dim_x * weight_dim_y;
  return IntegerSequenceCodec::
      GetBitCountForRange(num_weights, target_weight_range);
}

std::vector<unsigned int> InfillWeights(
    const std::vector<unsigned int>& weights, Footprint footprint,
    unsigned int dim_x, unsigned int dim_y) {
  std::vector<unsigned int> result;
  result.reserve(footprint.NumPixels());
  for (unsigned int t = 0; t < footprint.Height(); ++t) {
    for (unsigned int s = 0; s < footprint.Width(); ++s) {
      const auto grid_space_coords =
          GetGridSpaceCoordinates(footprint, s, t, dim_x, dim_y);
      const auto grid_pts =
          BilerpGridPointsForWeight(grid_space_coords, dim_x);
      const auto grid_factors =
          BilerpGridPointFactorsForWeight(grid_space_coords);

      unsigned int weight = 0;
      for (unsigned int i = 0; i < 4; ++i) {
        if (grid_pts[i] < dim_x * dim_y) {
          weight += weights.at(grid_pts[i]) * grid_factors[i];
        }
      }
      result.push_back((weight + 8) >> 4);
    }
  }

  return result;
}

}  // namespace astc_codec
