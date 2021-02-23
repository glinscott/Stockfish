/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Definition of layer AffineTransform of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include <iostream>
#include "../nnue_common.h"

namespace Eval::NNUE::Layers {

  // Affine transformation layer
  template <typename PreviousLayer, IndexType OutputDimensions, bool UseRelu>
  class AffineTransform {
    static constexpr bool kFastScaling = true;

#if defined(USE_AVX512) || defined(USE_AVX2)
    static constexpr bool kVectorized = UseRelu;
#else
    static constexpr bool kVectorized = false;
#endif
   public:
    // Input/output type
    using InputType = typename PreviousLayer::OutputType;
    using OutputType = typename std::conditional<UseRelu, std::uint8_t, std::int32_t>::type;
    static_assert(std::is_same<InputType, std::uint8_t>::value, "");

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions =
        PreviousLayer::kOutputDimensions;
    static constexpr IndexType kOutputDimensions = OutputDimensions;
    static constexpr IndexType kPaddedInputDimensions =
        CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

    // Size of forward propagation buffer used in this layer
    static constexpr std::size_t kSelfBufferSize =
        CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

    // Size of the forward propagation buffer used from the input layer to this layer
    static constexpr std::size_t kBufferSize =
        PreviousLayer::kBufferSize + kSelfBufferSize;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {
      std::uint32_t hash_value = 0xCC03DAE4u;
      hash_value += kOutputDimensions;
      hash_value ^= PreviousLayer::GetHashValue() >> 1;
      hash_value ^= PreviousLayer::GetHashValue() << 31;
      return hash_value;
    }

   // Read network parameters
    bool ReadParameters(std::istream& stream) {
      if (!previous_layer_.ReadParameters(stream)) return false;

      scale_ = read_little_endian<std::int32_t>(stream);
      scale_bits_ = read_little_endian<std::int32_t>(stream);
      if (UseRelu && kFastScaling) {
        scale_ >>= 16;
        scale_bits_ -= 16;
      }
      weight_zero_point_ = read_little_endian<std::int32_t>(stream);

      // Note: biases are technically 32 bit, but actually are about 9 bits.
      for (std::size_t i = 0; i < kOutputDimensions; ++i)
        biases_[i] = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < kOutputDimensions * kPaddedInputDimensions; ++i) {
        WeightType w = read_little_endian<WeightType>(stream);
        if (kVectorized)
          weights_[
            (i / 4) % (kPaddedInputDimensions / 4) * kOutputDimensions * 4 +
            i / kPaddedInputDimensions * 4 +
            i % 4
          ] = w;
        else
          weights_[i] = w;
      }

      return !stream.fail();
    }

    // Forward propagation
    const OutputType* Propagate(
        const TransformedFeatureType* transformed_features, char* buffer) const {
      const auto input = previous_layer_.Propagate(
          transformed_features, buffer + kSelfBufferSize);
      auto output = reinterpret_cast<OutputType*>(buffer);

#if defined(USE_AVX512) || defined(USE_AVX2)

      [[maybe_unused]] const __m256i kOnes256 = _mm256_set1_epi16(1);

      [[maybe_unused]] auto m256_add_dpbusd_epi32x4 = [=](__m256i& acc, __m256i a0, __m256i b0, __m256i a1, __m256i b1,
                                                                        __m256i a2, __m256i b2, __m256i a3, __m256i b3) {
        __m256i product0 = _mm256_maddubs_epi16(a0, b0);
        __m256i product1 = _mm256_maddubs_epi16(a1, b1);
        __m256i product2 = _mm256_maddubs_epi16(a2, b2);
        __m256i product3 = _mm256_maddubs_epi16(a3, b3);
        product0 = _mm256_add_epi16(product0, product1);
        product2 = _mm256_add_epi16(product2, product3);
        product0 = _mm256_add_epi16(product0, product2);
        product0 = _mm256_madd_epi16(product0, kOnes256);
        acc = _mm256_add_epi32(acc, product0);
      };

      [[maybe_unused]] auto m256_scale = [=](__m256i a0, __m256i a1, __m256i zero_point, __m256i scale) {
        __m256i sum0 = _mm256_sub_epi32(a0, zero_point);
        __m256i sum1 = _mm256_sub_epi32(a1, zero_point);
        sum0 = _mm256_mullo_epi32(sum0, scale);
        sum1 = _mm256_mullo_epi32(sum1, scale);
        sum0 = rounding_shift(sum0, 15);
        sum1 = rounding_shift(sum1, 15);
        __m256i packed = _mm256_packs_epi32(sum0, sum1);  // saturated 32bit -> 16bit, pack (a0, a1) together into one
        __m256i result = _mm256_permute4x64_epi64(packed, 0xD8); // permute [0,1,2,3] to [0,2,1,3]
        return result;
      };
#endif

#if defined(USE_AVX512) || defined(USE_AVX2)
      using vec_t = __m256i;
      #define vec_setzero _mm256_setzero_si256
      #define vec_set_32 _mm256_set1_epi32
      auto& vec_add_dpbusd_32x4 = m256_add_dpbusd_epi32x4;
      static constexpr const IndexType kOutputSimdWidth = kSimdWidth / 4;
      const __m256i kScale = _mm256_set1_epi32(scale_);
      constexpr int kControl = 0b11011000;
#endif

      std::int32_t zero_point_sum = 0;
      for (IndexType j = 0; j < kInputDimensions; ++j)
        zero_point_sum += input[j];
      zero_point_sum *= weight_zero_point_;

      if (kVectorized) {
#if defined(USE_AVX512) || defined(USE_AVX2)
          constexpr IndexType kNumChunks = kPaddedInputDimensions / 4;

          const auto input32 = reinterpret_cast<const std::int32_t*>(input);
          std::int32_t sums[kOutputDimensions];
          std::memcpy(sums, biases_, kOutputDimensions * sizeof(BiasType));
          vec_t *sumptr = reinterpret_cast<vec_t*>(sums);

          for (int i = 0; i < (int)kNumChunks - 3; i += 4)
          {
              const vec_t in0 = vec_set_32(input32[i + 0]);
              const vec_t in1 = vec_set_32(input32[i + 1]);
              const vec_t in2 = vec_set_32(input32[i + 2]);
              const vec_t in3 = vec_set_32(input32[i + 3]);
              const auto col0 = reinterpret_cast<const vec_t*>(&weights_[(i + 0) * kOutputDimensions * 4]);
              const auto col1 = reinterpret_cast<const vec_t*>(&weights_[(i + 1) * kOutputDimensions * 4]);
              const auto col2 = reinterpret_cast<const vec_t*>(&weights_[(i + 2) * kOutputDimensions * 4]);
              const auto col3 = reinterpret_cast<const vec_t*>(&weights_[(i + 3) * kOutputDimensions * 4]);
              for (int j = 0; j * kOutputSimdWidth < kOutputDimensions; ++j)
                  vec_add_dpbusd_32x4(sumptr[j], in0, col0[j], in1, col1[j], in2, col2[j], in3, col3[j]);
          }

          if (kFastScaling) {
            vec_t *outptr = reinterpret_cast<vec_t*>(output);
            vec_t zero_point_vec = vec_set_32(zero_point_sum);
            for (int j = 0; j * kSimdWidth < kOutputDimensions; j++) {
              __m256i sum0 = m256_scale(sumptr[j * 4 + 0], sumptr[j * 4 + 1], zero_point_vec, kScale);
              __m256i sum1 = m256_scale(sumptr[j * 4 + 2], sumptr[j * 4 + 3], zero_point_vec, kScale);
              _mm256_store_si256(&outptr[j], _mm256_permute4x64_epi64(
                  _mm256_packus_epi16(sum0, sum1), kControl));
            }
          } else {
            for (IndexType i = 0; i < kOutputDimensions; ++i) {
              std::int32_t sum = sums[i];
              sum = rounding_shift(static_cast<std::int64_t>(sum - zero_point_sum) * scale_, scale_bits_);
              sum = std::max(std::min(sum, 255), 0);
              output[i] = static_cast<OutputType>(sum);
            }
          }
#endif
      } else {
        for (IndexType i = 0; i < kOutputDimensions; ++i) {
          const IndexType offset = i * kPaddedInputDimensions;

          std::int32_t sum = biases_[i];
          for (IndexType j = 0; j < kInputDimensions; ++j) {
            sum += weights_[offset + j] * input[j];
          }
          sum -= zero_point_sum;
          if (UseRelu) {
            sum = rounding_shift(static_cast<std::int64_t>(sum) * scale_, scale_bits_);
            sum = std::max(std::min(sum, 255), 0);
          } else {
            sum = rounding_shift(static_cast<std::int64_t>(sum * 600) * scale_, scale_bits_);
          }
          output[i] = static_cast<OutputType>(sum);
        }
      }

      return output;
    }

   private:
    using BiasType = std::int32_t;
    using WeightType = std::int8_t;

    PreviousLayer previous_layer_;

    // Quantization parameters
    std::int32_t scale_;
    std::int32_t scale_bits_;
    std::int32_t weight_zero_point_;

    alignas(kCacheLineSize) BiasType biases_[kOutputDimensions];
    alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedInputDimensions];
  };

}  // namespace Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
