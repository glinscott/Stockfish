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
  template <typename PreviousLayer, IndexType OutputDimensions>
  class AffineTransform {
   public:
    // Input/output type
    using InputType = typename PreviousLayer::OutputType;
    using OutputType = float;
    static_assert(std::is_same<InputType, float>::value, "");

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
      for (std::size_t i = 0; i < kOutputDimensions; ++i)
        biases_[i] = _mm_set_ps(0.0f, 0.0f, 0.0f, read_little_endian<BiasType>(stream));
      for (std::size_t i = 0; i < kOutputDimensions * kPaddedInputDimensions; ++i)
        weights_[i] = read_little_endian<WeightType>(stream);
      return !stream.fail();
    }

    // Forward propagation
    const OutputType* Propagate(
        const TransformedFeatureType* transformed_features, char* buffer) const {
      const auto input = previous_layer_.Propagate(
          transformed_features, buffer + kSelfBufferSize);
      const auto output = reinterpret_cast<OutputType*>(buffer);

      if constexpr (kOutputDimensions % 4 == 0)
      {
        for (IndexType i = 0; i < kOutputDimensions; i += 4) {
          const IndexType offset0 = i * kPaddedInputDimensions;
          const IndexType offset1 = (i + 1) * kPaddedInputDimensions;
          const IndexType offset2 = (i + 2) * kPaddedInputDimensions;
          const IndexType offset3 = (i + 3) * kPaddedInputDimensions;
          auto sum0 = biases_[i];
          auto sum1 = biases_[i+1];
          auto sum2 = biases_[i+2];
          auto sum3 = biases_[i+3];
          for (IndexType j = 0; j < kInputDimensions; j += 4) {
            sum0 = _mm_add_ps(sum0, _mm_mul_ps(*reinterpret_cast<const __m128*>(&weights_[offset0 + j]), *reinterpret_cast<const __m128*>(&input[j])));
            sum1 = _mm_add_ps(sum1, _mm_mul_ps(*reinterpret_cast<const __m128*>(&weights_[offset1 + j]), *reinterpret_cast<const __m128*>(&input[j])));
            sum2 = _mm_add_ps(sum2, _mm_mul_ps(*reinterpret_cast<const __m128*>(&weights_[offset2 + j]), *reinterpret_cast<const __m128*>(&input[j])));
            sum3 = _mm_add_ps(sum3, _mm_mul_ps(*reinterpret_cast<const __m128*>(&weights_[offset3 + j]), *reinterpret_cast<const __m128*>(&input[j])));
          }
          *reinterpret_cast<__m128*>(&output[i]) = _mm_hadd_ps(
            _mm_hadd_ps(sum0, sum1),
            _mm_hadd_ps(sum2, sum3)
          );
        }
      }
      else
      {
        for (IndexType i = 0; i < kOutputDimensions; ++i) {
          const IndexType offset = i * kPaddedInputDimensions;
          OutputType sum = *reinterpret_cast<const float*>(&biases_[i]);
          for (IndexType j = 0; j < kInputDimensions; ++j) {
            sum += weights_[offset + j] * input[j];
          }
          output[i] = sum;
        }
      }

      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = float;

    PreviousLayer previous_layer_;

    alignas(kCacheLineSize) __m128 biases_[kOutputDimensions];
    alignas(kCacheLineSize) WeightType weights_[kOutputDimensions * kPaddedInputDimensions];
  };

}  // namespace Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
