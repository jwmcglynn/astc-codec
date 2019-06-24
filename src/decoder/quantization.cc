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

#include "src/decoder/quantization.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <memory>
#include <vector>

#include "src/base/math_utils.h"

namespace astc_codec {

namespace {

// Trit unquantization procedure as described in Section C.2.13
unsigned int GetUnquantizedTritValue(unsigned int trit, unsigned int bits,
                                     unsigned int range) {
  unsigned int a = (bits & 1) ? 0x1FF : 0;
  unsigned int b = 0, c = 0;
  switch (range) {
    case 5: {
      b = 0;
      c = 204;
    } break;

    case 11: {
      unsigned int x = (bits >> 1) & 0x1;
      b = (x << 1) | (x << 2) | (x << 4) | (x << 8);
      c = 93;
    } break;

    case 23: {
      unsigned int x = (bits >> 1) & 0x3;
      b = x | (x << 2) | (x << 7);
      c = 44;
    } break;

    case 47: {
      unsigned int x = (bits >> 1) & 0x7;
      b = x | (x << 6);
      c = 22;
    } break;

    case 95: {
      unsigned int x = (bits >> 1) & 0xF;
      b = (x >> 2) | (x << 5);
      c = 11;
    } break;

    case 191: {
      unsigned int x = (bits >> 1) & 0x1F;
      b = (x >> 4) | (x << 4);
      c = 5;
    } break;

    default:
      assert(false && "Illegal trit encoding");
      break;
  }

  unsigned int t = trit * c + b;
  t ^= a;
  t = (a & 0x80) | (t >> 2);
  return t;
}

// Quint unquantization procedure as described in Section C.2.13
unsigned int GetUnquantizedQuintValue(unsigned int quint, unsigned int bits,
                                      unsigned int range) {
  unsigned int a = (bits & 1) ? 0x1FF : 0;
  unsigned int b = 0, c = 0;
  switch (range) {
    case 9: {
      b = 0;
      c = 113;
    } break;

    case 19: {
      unsigned int x = (bits >> 1) & 0x1;
      b = (x << 2) | (x << 3) | (x << 8);
      c = 54;
    } break;

    case 39: {
      unsigned int x = (bits >> 1) & 0x3;
      b = (x >> 1) | (x << 1) | (x << 7);
      c = 26;
    } break;

    case 79: {
      unsigned int x = (bits >> 1) & 0x7;
      b = (x >> 1) | (x << 6);
      c = 13;
    } break;

    case 159: {
      unsigned int x = (bits >> 1) & 0xF;
      b = (x >> 3) | (x << 5);
      c = 6;
    } break;

    default:
      assert(false && "Illegal quint encoding");
      break;
  }

  unsigned int t = quint * c + b;
  t ^= a;
  t = (a & 0x80) | (t >> 2);
  return t;
}

// Trit unquantization procedure as described in Section C.2.17. In the code
// below, the variables a, b, and c correspond to the columns A, B, and C in
// the specification.
unsigned int GetUnquantizedTritWeight(unsigned int trit, unsigned int bits,
                                      unsigned int range) {
  unsigned int a = (bits & 1) ? 0x7F : 0;
  unsigned int b = 0, c = 0;
  switch (range) {
    case 2: return (std::array<unsigned int, 3>{{0, 32, 63}})[trit];

    case 5: {
      c = 50;
      b = 0;
    } break;

    case 11: {
      c = 23;
      b = (bits >> 1) & 1;
      b |= (b << 2) | (b << 6);
    }
    break;

    case 23: {
      c = 11;
      b = (bits >> 1) & 0x3;
      b |= (b << 5);
    }
    break;

    default:
      assert(false && "Illegal trit encoding");
      break;
  }

  unsigned int t = trit * c + b;
  t ^= a;
  t = (a & 0x20) | (t >> 2);
  return t;
}

// Quint unquantization procedure as described in Section C.2.17. In the code
// below, the variables a, b, and c correspond to the columns A, B, and C in
// the specification.
unsigned int GetUnquantizedQuintWeight(unsigned int quint, unsigned int bits,
                                       unsigned int range) {
  unsigned int a = (bits & 1) ? 0x7F : 0;
  unsigned int b = 0, c = 0;
  switch (range) {
    case 4: return (std::array<unsigned int, 5>{{0, 16, 32, 47, 63}})[quint];

    case 9: {
      c = 28;
      b = 0;
    } break;

    case 19: {
      c = 13;
      b = (bits >> 1) & 0x1;
      b = (b << 1) | (b << 6);
    }
    break;

    default:
      assert(false && "Illegal quint encoding");
      break;
  }

  unsigned int t = quint * c + b;
  t ^= a;
  t = (a & 0x20) | (t >> 2);
  return t;
}

// A Quantization map allows us to convert to/from values that are quantized
// according to the ASTC spec.
class QuantizationMap {
 public:
  unsigned int Quantize(unsigned int x) const {
    return x < quantization_map_.size() ? quantization_map_.at(x) : 0;
  }

  unsigned int Unquantize(unsigned int x) const {
    return x < unquantization_map_.size() ? unquantization_map_.at(x) : 0;
  }

 protected:
  QuantizationMap() { }
  std::vector<unsigned int> quantization_map_;
  std::vector<unsigned int> unquantization_map_;

  void GenerateQuantizationMap() {
    assert(unquantization_map_.size() > 1);
    quantization_map_.clear();

    // TODO(google) For weights, we don't need quantization values all the
    // way up to 256, but it doesn't hurt -- just wastes memory, but the code
    // is much cleaner this way
    for (unsigned int i = 0; i < 256; ++i) {
      unsigned int best_idx = 0;
      unsigned int best_idx_score = 256;
      unsigned int idx = 0;
      for (unsigned int unquantized_val : unquantization_map_) {
        const int diff =
            static_cast<int>(i) - static_cast<int>(unquantized_val);
        const unsigned int idx_score = static_cast<unsigned int>(diff * diff);
        if (idx_score < best_idx_score) {
          best_idx = idx;
          best_idx_score = idx_score;
        }
        idx++;
      }

      quantization_map_.push_back(best_idx);
    }
  }
};

template <unsigned int (*UnquantizationFunc)(unsigned int, unsigned int,
                                             unsigned int)>
class TritQuantizationMap : public QuantizationMap {
 public:
  explicit TritQuantizationMap(unsigned int range) : QuantizationMap() {
    assert((range + 1) % 3 == 0);
    const unsigned int num_bits_pow_2 = (range + 1) / 3;
    const unsigned int num_bits =
        num_bits_pow_2 == 0
            ? 0
            : static_cast<unsigned int>(base::Log2Floor(num_bits_pow_2));

    for (unsigned int trit = 0; trit < 3; ++trit) {
      for (unsigned int bits = 0; bits < (1u << num_bits); ++bits) {
        unquantization_map_.push_back(UnquantizationFunc(trit, bits, range));
      }
    }

    GenerateQuantizationMap();
  }
};

template <unsigned int (*UnquantizationFunc)(unsigned int, unsigned int,
                                             unsigned int)>
class QuintQuantizationMap : public QuantizationMap {
 public:
  explicit QuintQuantizationMap(unsigned int range) : QuantizationMap() {
    assert((range + 1) % 5 == 0);
    const unsigned int num_bits_pow_2 = (range + 1) / 5;
    const unsigned int num_bits =
        num_bits_pow_2 == 0
            ? 0
            : static_cast<unsigned int>(base::Log2Floor(num_bits_pow_2));

    for (unsigned int quint = 0; quint < 5; ++quint) {
      for (unsigned int bits = 0; bits < (1u << num_bits); ++bits) {
        unquantization_map_.push_back(UnquantizationFunc(quint, bits, range));
      }
    }

    GenerateQuantizationMap();
  }
};

template <unsigned int TotalUnquantizedBits>
class BitQuantizationMap : public QuantizationMap {
 public:
  explicit BitQuantizationMap<TotalUnquantizedBits>(unsigned int range)
      : QuantizationMap() {
    // Make sure that if we're using bits then we have a positive power of two.
    assert(base::CountOnes(range + 1) == 1);

    const unsigned int num_bits =
        static_cast<unsigned int>(base::Log2Floor(range + 1));
    for (unsigned int bits = 0; bits <= range; ++bits) {
      // Need to replicate bits until we fill up the bits
      unsigned int unquantized = bits;
      unsigned int num_unquantized_bits = num_bits;
      while (num_unquantized_bits < TotalUnquantizedBits) {
        const unsigned int num_dst_bits_to_shift_up =
            std::min(num_bits, TotalUnquantizedBits - num_unquantized_bits);
        const unsigned int num_src_bits_to_shift_down =
            num_bits - num_dst_bits_to_shift_up;
        unquantized <<= num_dst_bits_to_shift_up;
        unquantized |= bits >> num_src_bits_to_shift_down;
        num_unquantized_bits += num_dst_bits_to_shift_up;
      }
      assert(num_unquantized_bits == TotalUnquantizedBits);

      unquantization_map_.push_back(unquantized);

      // Fill half of the quantization map with the previous value for bits
      // and the other half with the current value for bits
      if (bits > 0) {
        const unsigned int prev_unquant = unquantization_map_.at(bits - 1);
        while (quantization_map_.size() <= (prev_unquant + unquantized) / 2) {
          quantization_map_.push_back(bits - 1);
        }
      }
      while (quantization_map_.size() <= unquantized) {
        quantization_map_.push_back(bits);
      }
    }

    assert(quantization_map_.size() == 1 << TotalUnquantizedBits);
  }
};

using QMap = std::shared_ptr<QuantizationMap>;

// Returns the quantization map for quantizing color values in [0, 255] with the
// smallest range that can accommodate |r|
static const QuantizationMap* GetQuantMapForValueRange(unsigned int r) {
  // Endpoint values can be quantized using bits, trits, or quints. Here we
  // store the quantization maps for each of the ranges that are supported by
  // such an encoding. That way we can choose the proper quantization procedure
  // based on the range of values rather than by having complicated switches and
  // logic. We must use a std::map here instead of a std::unordered_map because
  // of the assumption made in std::upper_bound about the iterators being from a
  // poset.
  static const auto* const kASTCEndpointQuantization = new std::map<
      unsigned int, QMap>{
      {5, QMap(new TritQuantizationMap<GetUnquantizedTritValue>(5))},        //
      {7, QMap(new BitQuantizationMap<8>(7))},                               //
      {9, QMap(new QuintQuantizationMap<GetUnquantizedQuintValue>(9))},      //
      {11, QMap(new TritQuantizationMap<GetUnquantizedTritValue>(11))},      //
      {15, QMap(new BitQuantizationMap<8>(15))},                             //
      {19, QMap(new QuintQuantizationMap<GetUnquantizedQuintValue>(19))},    //
      {23, QMap(new TritQuantizationMap<GetUnquantizedTritValue>(23))},      //
      {31, QMap(new BitQuantizationMap<8>(31))},                             //
      {39, QMap(new QuintQuantizationMap<GetUnquantizedQuintValue>(39))},    //
      {47, QMap(new TritQuantizationMap<GetUnquantizedTritValue>(47))},      //
      {63, QMap(new BitQuantizationMap<8>(63))},                             //
      {79, QMap(new QuintQuantizationMap<GetUnquantizedQuintValue>(79))},    //
      {95, QMap(new TritQuantizationMap<GetUnquantizedTritValue>(95))},      //
      {127, QMap(new BitQuantizationMap<8>(127))},                           //
      {159, QMap(new QuintQuantizationMap<GetUnquantizedQuintValue>(159))},  //
      {191, QMap(new TritQuantizationMap<GetUnquantizedTritValue>(191))},    //
      {255, QMap(new BitQuantizationMap<8>(255))},                           //
  };

  assert(r < 256);
  auto itr = kASTCEndpointQuantization->upper_bound(r);
  if (itr != kASTCEndpointQuantization->begin()) {
    return (--itr)->second.get();
  }
  return nullptr;
}

// Returns the quantization map for weight values in [0, 63] with the smallest
// range that can accommodate |r|
static const QuantizationMap* GetQuantMapForWeightRange(unsigned int r) {
  // Similar to endpoint quantization, weights can also be stored using trits,
  // quints, or bits. Here we store the quantization maps for each of the ranges
  // that are supported by such an encoding.
  static const auto* const kASTCWeightQuantization = new std::map<unsigned int,
                                                                  QMap>{
      {1, QMap(new BitQuantizationMap<6>(1))},                              //
      {2, QMap(new TritQuantizationMap<GetUnquantizedTritWeight>(2))},      //
      {3, QMap(new BitQuantizationMap<6>(3))},                              //
      {4, QMap(new QuintQuantizationMap<GetUnquantizedQuintWeight>(4))},    //
      {5, QMap(new TritQuantizationMap<GetUnquantizedTritWeight>(5))},      //
      {7, QMap(new BitQuantizationMap<6>(7))},                              //
      {9, QMap(new QuintQuantizationMap<GetUnquantizedQuintWeight>(9))},    //
      {11, QMap(new TritQuantizationMap<GetUnquantizedTritWeight>(11))},    //
      {15, QMap(new BitQuantizationMap<6>(15))},                            //
      {19, QMap(new QuintQuantizationMap<GetUnquantizedQuintWeight>(19))},  //
      {23, QMap(new TritQuantizationMap<GetUnquantizedTritWeight>(23))},    //
      {31, QMap(new BitQuantizationMap<6>(31))},                            //
  };

  assert(r < 32);
  auto itr = kASTCWeightQuantization->upper_bound(r);
  if (itr != kASTCWeightQuantization->begin()) {
    return (--itr)->second.get();
  }
  return nullptr;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////

unsigned int QuantizeCEValueToRange(unsigned int value,
                                    unsigned int range_max_value) {
  assert(range_max_value >= kEndpointRangeMinValue);
  assert(range_max_value <= 255);
  assert(value <= 255);

  const QuantizationMap* map = GetQuantMapForValueRange(range_max_value);
  return map ? map->Quantize(value) : 0;
}

unsigned int UnquantizeCEValueFromRange(unsigned int value,
                                        unsigned int range_max_value) {
  assert(range_max_value >= kEndpointRangeMinValue);
  assert(range_max_value <= 255);
  assert(value <= range_max_value);

  const QuantizationMap* map = GetQuantMapForValueRange(range_max_value);
  return map ? map->Unquantize(value) : 0;
}

unsigned int QuantizeWeightToRange(unsigned int weight,
                                   unsigned int range_max_value) {
  assert(range_max_value >= 1);
  assert(range_max_value <= kWeightRangeMaxValue);
  assert(weight <= 64);

  // The quantization maps that define weight unquantization expect values in
  // the range [0, 64), but the specification quantizes them to the range
  // [0, 64] according to C.2.17. This is a slight hack similar to the one in
  // the unquantization procedure to return the passed in unquantized value to
  // [0, 64) prior to running it through the quantization procedure.
  if (weight > 33) {
    weight -= 1;
  }
  const QuantizationMap* map = GetQuantMapForWeightRange(range_max_value);
  return map ? map->Quantize(weight) : 0;
}

unsigned int UnquantizeWeightFromRange(unsigned int weight,
                                       unsigned int range_max_value) {
  assert(range_max_value >= 1);
  assert(range_max_value <= kWeightRangeMaxValue);
  assert(weight <= range_max_value);
  const QuantizationMap* map = GetQuantMapForWeightRange(range_max_value);
  unsigned int dq = map ? map->Unquantize(weight) : 0;

  // Quantized weights are returned in the range [0, 64), but they should be
  // returned in the range [0, 64], so according to C.2.17 we need to add one
  // to the result.
  assert(dq < 64);
  if (dq > 32) {
    dq += 1;
  }
  return dq;
}

}  // namespace astc_codec
