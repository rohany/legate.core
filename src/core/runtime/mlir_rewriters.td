//===- mlir_rewriters.td -*- tablegen -*-===//

#ifndef MLIR_REWRITERS
#define MLIR_REWRITERS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

def ZeroFLikeMatcher : NativeCodeCall<"::mlir::success("
    "::mlir::matchPattern($_self->getResult(0), ::mlir::m_AnyZeroFloat()))">;

def OneILikeMatcher : NativeCodeCall<"::mlir::success("
    "::mlir::matchPattern($_self->getResult(0), ::mlir::m_One()))">;

// Define normalization patterns for reduction operations
// against their identities. Inspection against the
// AtomicRMWKind enum in MLIR and the corresponding
// definitions in ArithOps.cc and ArithCanonicalization.td
// yeilds the following missing rewrite rules.

// AddFZeroLHS add(a, 0) -> a
def AddFZeroLHS :
    Pat<(Arith_AddFOp $a, (ZeroFLikeMatcher $c), $fmf),
        (replaceWithValue $a)>;

// AddFZeroRHS add(0, a) -> a
def AddFZeroRHS :
    Pat<(Arith_AddFOp (ZeroFLikeMatcher $c), $a, $fmf),
        (replaceWithValue $a)>;

// AndIOneLHS and(a, 1) -> a
def AndIOneLHS :
    Pat<(Arith_AndIOp $a, (OneILikeMatcher $c)),
        (replaceWithValue $a)>;

// AndIOneRHS and(1, a) -> a
def AndIOneRHS :
    Pat<(Arith_AndIOp (OneILikeMatcher $c), $a),
        (replaceWithValue $a)>;

#endif // MLIR_REWRITERS
