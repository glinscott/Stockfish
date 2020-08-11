#ifndef _NNUE_FEATURES_MOBILITY_H_
#define _NNUE_FEATURES_MOBILITY_H_

#include "../../evaluate.h"
#include "features_common.h"

namespace Eval {

namespace NNUE {

namespace Features {

// Mobility
//
// Encoded as a bitboard per piece type.  Mobility is "raw" mobility - we rely
// on the net to interpret "safe".  Supports incremental computation.
class Mobility {
 public:
  // feature quantity name
  static constexpr const char* kName = "Mobility";
  // Hash value embedded in the evaluation function file
  static constexpr std::uint32_t kHashValue = 0x62924F91u;
  // number of feature dimensions (bitboard per piece type)
  static constexpr IndexType kDimensions = 64 * 6 * 2;
  // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
  static constexpr IndexType kMaxActiveDimensions = 32 * 6 * 2;
  // Timing of full calculation instead of difference calculation
  static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kNone;

  // Get a list of indices with a value of 1 among the features
  static void AppendActiveIndices(const Position& pos, Color perspective,
                                  IndexList* active);

  // Get a list of indices whose values have changed from the previous one in the feature quantity
  static void AppendChangedIndices(const Position& pos, Color perspective,
                                   IndexList* removed, IndexList* added);
};

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif
