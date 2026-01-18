#include "toy/MLIRGen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

namespace toy {

class MLIRGenImpl {
 public:
  explicit MLIRGenImpl(mlir::MLIRContext& context)
      : context(context), builder(&context) {}

  mlir::ModuleOp generate(ExprAST& ast) {
    module = mlir::ModuleOp::create(builder.getUnknownLoc());

    // Create a main function: func.func @main() -> f64
    auto doubleType = builder.getF64Type();
    auto funcType = builder.getFunctionType({}, {doubleType});
    auto mainFunc =
        mlir::func::FuncOp::create(builder.getUnknownLoc(), "main", funcType);

    auto& entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    mlir::Value result = codegen(ast);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), result);

    module.push_back(mainFunc);
    return module;
  }

 private:
  mlir::Value codegen(ExprAST& node) {
    if (auto* num = dynamic_cast<NumberExprAST*>(&node)) {
      return builder.create<mlir::arith::ConstantOp>(
          builder.getUnknownLoc(), builder.getF64FloatAttr(num->getVal()));
    }

    if (auto* bin = dynamic_cast<BinaryExprAST*>(&node)) {
      mlir::Value lhs = codegen(*bin->getLHS());
      mlir::Value rhs = codegen(*bin->getRHS());
      switch (bin->getOp()) {
        case '+':
          return builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '-':
          return builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '*':
          return builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '/':
          return builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
      }
    }
    return nullptr;
  }

  mlir::MLIRContext& context;
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
};

mlir::ModuleOp mlirGen(mlir::MLIRContext& context, ExprAST& ast) {
  return MLIRGenImpl(context).generate(ast);
}

}  // namespace toy
