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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <cstring> // std::memset()

namespace Eval::NNUE {

  // Input feature converter
  class FeatureTransformer {

   private:
    // Number of output dimensions for one side
    static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions = RawFeatures::kDimensions;
    static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;

    // Size of forward propagation buffer
    static constexpr std::size_t kBufferSize =
        kOutputDimensions * sizeof(OutputType);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {

      return RawFeatures::kHashValue ^ kOutputDimensions;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {

      for (std::size_t i = 0; i < kHalfDimensions; ++i)
        biases_[i] = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
        weights_[i] = read_little_endian<WeightType>(stream);
      return !stream.fail();
    }

    // Convert input features
    void Transform(const Position& pos, OutputType* output) const {

      UpdateAccumulator(pos, WHITE);
      UpdateAccumulator(pos, BLACK);

      const auto& accumulation = pos.state()->accumulator.accumulation;

  #if defined(USE_AVX2)
      constexpr IndexType kChunkSize = kSimdWidth / sizeof(TransformedFeatureType);
      constexpr IndexType kNumChunks = kHalfDimensions / kChunkSize;
      const __m256 kZero = _mm256_setzero_ps();
      const __m256 kOne = _mm256_set1_ps(1.0f);

  #elif defined(USE_SSE2)
      constexpr IndexType kChunkSize = kSimdWidth / sizeof(TransformedFeatureType);
      constexpr IndexType kNumChunks = kHalfDimensions / kChunkSize;
      const __m128 kZero = _mm_setzero_ps();
      const __m128 kOne = _mm_set1_ps(1.0f);

  #elif defined(USE_MMX)
      static_assert(false);

  #elif defined(USE_NEON)
      static_assert(false);
  #endif

      const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
      for (IndexType p = 0; p < 2; ++p) {
        const IndexType offset = kHalfDimensions * p;

  #if defined(USE_AVX2)
        auto out = reinterpret_cast<__m256*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          __m256 sum = reinterpret_cast<const __m256*>(accumulation[perspectives[p]][0])[j];
          out[j] = _mm256_min_ps(_mm256_max_ps(sum, kZero), kOne);
        }

  #elif defined(USE_SSE2)
        auto out = reinterpret_cast<__m128*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          __m128 sum = reinterpret_cast<const __m128*>(accumulation[perspectives[p]][0])[j];

          out[j] = _mm_min_ps(_mm_max_ps(sum, kZero), kOne);
        }
  #endif

      }
    }

   private:
    void UpdateAccumulator(const Position& pos, const Color c) const {

      // Look for a usable accumulator of an earlier position. We keep track
      // of the estimated gain in terms of features to be added/subtracted.
      StateInfo *st = pos.state(), *next = nullptr;
      int gain = pos.count<ALL_PIECES>() - 2;
      while (st->accumulator.state[c] == EMPTY)
      {
        auto& dp = st->dirtyPiece;
        // The first condition tests whether an incremental update is
        // possible at all: if this side's king has moved, it is not possible.
        static_assert(std::is_same_v<RawFeatures::SortedTriggerSet,
              Features::CompileTimeList<Features::TriggerEvent, Features::TriggerEvent::kFriendKingMoved>>,
              "Current code assumes that only kFriendlyKingMoved refresh trigger is being used.");
        if (   dp.piece[0] == make_piece(c, KING)
            || (gain -= dp.dirty_num + 1) < 0)
          break;
        next = st;
        st = st->previous;
      }

      if (st->accumulator.state[c] == COMPUTED)
      {
        if (next == nullptr)
          return;

        // Update incrementally in two steps. First, we update the "next"
        // accumulator. Then, we update the current accumulator (pos.state()).

        // Gather all features to be updated. This code assumes HalfKP features
        // only and doesn't support refresh triggers.
        static_assert(std::is_same_v<Features::FeatureSet<Features::HalfKP<Features::Side::kFriend>>,
                                     RawFeatures>);
        Features::IndexList removed[2], added[2];
        Features::HalfKP<Features::Side::kFriend>::AppendChangedIndices(pos,
            next->dirtyPiece, c, &removed[0], &added[0]);
        for (StateInfo *st2 = pos.state(); st2 != next; st2 = st2->previous)
          Features::HalfKP<Features::Side::kFriend>::AppendChangedIndices(pos,
              st2->dirtyPiece, c, &removed[1], &added[1]);

        // Mark the accumulators as computed.
        next->accumulator.state[c] = COMPUTED;
        pos.state()->accumulator.state[c] = COMPUTED;

        // Now update the accumulators listed in info[], where the last element is a sentinel.
        StateInfo *info[3] =
          { next, next == pos.state() ? nullptr : pos.state(), nullptr };

        for (IndexType i = 0; info[i]; ++i)
        {
          std::memcpy(info[i]->accumulator.accumulation[c][0],
              st->accumulator.accumulation[c][0],
              kHalfDimensions * sizeof(BiasType));
          st = info[i];

          // Difference calculation for the deactivated features
          for (const auto index : removed[i])
          {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              st->accumulator.accumulation[c][0][j] -= weights_[offset + j];
          }

          // Difference calculation for the activated features
          for (const auto index : added[i])
          {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              st->accumulator.accumulation[c][0][j] += weights_[offset + j];
          }
        }
      }
      else
      {
        // Refresh the accumulator
        auto& accumulator = pos.state()->accumulator;
        accumulator.state[c] = COMPUTED;
        Features::IndexList active;
        Features::HalfKP<Features::Side::kFriend>::AppendActiveIndices(pos, c, &active);

        std::memcpy(accumulator.accumulation[c][0], biases_,
            kHalfDimensions * sizeof(BiasType));

        for (const auto index : active)
        {
          const IndexType offset = kHalfDimensions * index;

          for (IndexType j = 0; j < kHalfDimensions; ++j)
            accumulator.accumulation[c][0][j] += weights_[offset + j];
        }
      }
    }

    using BiasType = float;
    using WeightType = float;

    alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
    alignas(kCacheLineSize)
        WeightType weights_[kHalfDimensions * kInputDimensions];
  };

}  // namespace Eval::NNUE

#endif // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
