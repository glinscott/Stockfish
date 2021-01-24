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

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED

#include "../nnue_common.h"

namespace Eval::NNUE::Layers {

  // Clipped ReLU
  template <typename PreviousLayer>
  class ClippedReLU {
   public:
    // Input/output type
    using InputType = typename PreviousLayer::OutputType;
    using OutputType = float;
    static_assert(std::is_same<InputType, float>::value, "");

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions =
        PreviousLayer::kOutputDimensions;
    static constexpr IndexType kOutputDimensions = kInputDimensions;

    // Size of forward propagation buffer used in this layer
    static constexpr std::size_t kSelfBufferSize =
        CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

    // Size of the forward propagation buffer used from the input layer to this layer
    static constexpr std::size_t kBufferSize =
        PreviousLayer::kBufferSize + kSelfBufferSize;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {
      std::uint32_t hash_value = 0x538D24C7u;
      hash_value += PreviousLayer::GetHashValue();
      return hash_value;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {
      return previous_layer_.ReadParameters(stream);
    }

    // Forward propagation
    const OutputType* Propagate(
        const TransformedFeatureType* transformed_features, char* buffer) const {
      const auto input = previous_layer_.Propagate(
          transformed_features, buffer + kSelfBufferSize);
      const auto output = reinterpret_cast<OutputType*>(buffer);

      constexpr IndexType kChunkSize = kSimdWidth / sizeof(InputType);
      constexpr IndexType kNumChunks = kInputDimensions / kChunkSize;

  #if defined(USE_AVX2)
      const __m256 kZero = _mm256_setzero_ps();
      const __m256 kOne = _mm256_set1_ps(1.0f);

      const auto in = reinterpret_cast<const __m256*>(input);
      const auto out = reinterpret_cast<__m256*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m256 v = in[i];
        out[i] = _mm256_min_ps(_mm256_max_ps(v, kZero), kOne);
      }

  #elif defined(USE_SSE2)
      const __m128 kZero = _mm_setzero_ps();
      const __m128 kOne = _mm_set1_ps(1.0f);

      const auto in = reinterpret_cast<const __m128*>(input);
      const auto out = reinterpret_cast<__m128*>(output);
      for (IndexType i = 0; i < kNumChunks; ++i) {
        const __m128 v = in[i];
        out[i] = _mm_min_ps(_mm_max_ps(v, kZero), kOne);
      }
  #endif

      return output;
    }

   private:
    PreviousLayer previous_layer_;
  };

}  // namespace Eval::NNUE::Layers

#endif // NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
