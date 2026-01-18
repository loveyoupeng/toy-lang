#include "toy/AST.h"
#include "toy/MLIRGen.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <memory>

int main(int argc, char **argv) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    // Manual AST for "1.0 + 1.0"
    auto lhs = std::make_unique<toy::NumberExprAST>(1.0);
    auto rhs = std::make_unique<toy::NumberExprAST>(1.0);
    auto binOp = std::make_unique<toy::BinaryExprAST>('+', std::move(lhs), std::move(rhs));

    auto module = toy::mlirGen(context, *binOp);

    if (!module) {
        llvm::errs() << "Failed to generate MLIR\n";
        return 1;
    }

    module->print(llvm::outs());

    return 0;
}