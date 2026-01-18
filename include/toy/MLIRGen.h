#ifndef TOY_MLIRGEN_H_
#define TOY_MLIRGEN_H_

#include "mlir/IR/BuiltinOps.h"
#include "toy/AST.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
}  // namespace mlir

namespace toy {

mlir::ModuleOp mlirGen(mlir::MLIRContext& context, ExprAST& ast);

}  // namespace toy

#endif  // TOY_MLIRGEN_H_
