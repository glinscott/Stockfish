#include "mobility.h"
#include "index_list.h"

namespace Eval {

namespace NNUE {

namespace Features {

void addIndex(Color Us, Bitboard b, int base, IndexList* active) {
  while (b) {
    Square s = pop_lsb(&b);
    if (Us == BLACK)
      s = rotate180(s);
    active->push_back(base + s);
  }
}

// Get a list of indices with a value of 1 among the features
void Mobility::AppendActiveIndices(
    const Position& pos, Color perspective, IndexList* active) {
  // do nothing if array size is small to avoid compiler warning
  if (RawFeatures::kMaxActiveDimensions < kMaxActiveDimensions) return;

  int base = 0;
  for (PieceType pt = PAWN; pt <= KING; ++pt) {
    Bitboard b = pos.state()->mobility[perspective][pt - 1];
    addIndex(perspective, b, base, active);
    base += 64;
  }
  Color them = ~perspective;
  for (PieceType pt = PAWN; pt <= KING; ++pt) {
    Bitboard b = pos.state()->mobility[them][pt - 1];
    addIndex(them, b, base, active);
    base += 64;
  }
}

// Get a list of indices whose values have changed from the previous one in the feature quantity
void Mobility::AppendChangedIndices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added) {
  const StateInfo* st = pos.state()->previous;

  int add = added->size(), rem = removed->size();

  int base = 0;
  for (PieceType pt = PAWN; pt <= KING; ++pt) {
    Bitboard b = pos.state()->mobility[perspective][pt - 1] ^ st->mobility[perspective][pt - 1];
    addIndex(perspective, b & pos.state()->mobility[perspective][pt - 1], base, added);
    addIndex(perspective, b & st->mobility[perspective][pt - 1], base, removed);
    base += 64;
  }
  Color them = ~perspective;
  for (PieceType pt = PAWN; pt <= KING; ++pt) {
    Bitboard b = pos.state()->mobility[them][pt - 1] ^ st->mobility[them][pt - 1];
    addIndex(them, b & pos.state()->mobility[them][pt - 1], base, added);
    addIndex(them, b & st->mobility[them][pt - 1], base, removed);
    base += 64;
  }
  add = added->size() - add;
  rem = removed->size() - rem;
  // printf("add %d rem %d tot %d\n", add, rem, add + rem);
}

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval
