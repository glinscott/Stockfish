#if defined(EVAL_NNUE)

#include "mobility.h"
#include "index_list.h"

namespace Eval {

namespace NNUE {

namespace Features {

Bitboard compress(Bitboard b) {
  Bitboard compressed = 0;
  while (b) {
    Square s = pop_lsb(&b);
    compressed |= 1 << ((int(file_of(s)) / 2) * 4 + int(rank_of(s)) / 2);
  }
  return compressed;
}
void addIndex(Color Us, Bitboard b, int base, IndexList* active) {
  while (b) {
    int s = int(pop_lsb(&b));
    if (Us == BLACK)
      s = 15 - s;
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
    addIndex(perspective, compress(b), base, active);
    base += 16;
  }
  Color them = ~perspective;
  for (PieceType pt = PAWN; pt <= KING; ++pt) {
    Bitboard b = pos.state()->mobility[them][pt - 1];
    addIndex(them, compress(b), base, active);
    base += 16;
  }
}

// Get a list of indices whose values have changed from the previous one in the feature quantity
void Mobility::AppendChangedIndices(
    const Position& pos, Color perspective,
    IndexList* removed, IndexList* added) {
  const StateInfo* st = pos.state()->previous;

  int base = 0;
  for (PieceType pt = PAWN; pt <= KING; ++pt) {
    Bitboard after = compress(pos.state()->mobility[perspective][pt - 1]);
    Bitboard before = compress(st->mobility[perspective][pt - 1]);
    Bitboard diff = before ^ after;
    addIndex(perspective, diff & after, base, added);
    addIndex(perspective, diff & before, base, removed);
    base += 16;
  }
  Color them = ~perspective;
  for (PieceType pt = PAWN; pt <= KING; ++pt) {
    Bitboard after = compress(pos.state()->mobility[them][pt - 1]);
    Bitboard before = compress(st->mobility[them][pt - 1]);
    Bitboard diff = before ^ after;
    addIndex(them, diff & after, base, added);
    addIndex(them, diff & before, base, removed);
    base += 16;
  }
}

}  // namespace Features

}  // namespace NNUE

}  // namespace Eval

#endif  // defined(EVAL_NNUE)
