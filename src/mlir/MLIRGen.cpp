#include "toy/MLIRGen.h"

#include <map>
#include <string>

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

    auto funcType = builder.getFunctionType({}, {});
    auto mainFunc =
        mlir::func::FuncOp::create(builder.getUnknownLoc(), "main", funcType);

    auto& entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    codegen(ast);

    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    module.push_back(mainFunc);
    return module;
  }

 private:
  mlir::Type getMLIRType(DataType type) {
    switch (type) {
      case DataType::Byte:
      case DataType::UInt8:
        return builder.getIntegerType(8, false);
      case DataType::Int8:
        return builder.getIntegerType(8);
      case DataType::Int16:
        return builder.getIntegerType(16);
      case DataType::Int32:
        return builder.getIntegerType(32);
      case DataType::Int64:
        return builder.getIntegerType(64);
      case DataType::UInt32:
        return builder.getIntegerType(32, false);
      case DataType::UInt64:
        return builder.getIntegerType(64, false);
      case DataType::Float32:
        return builder.getF32Type();
      case DataType::Float64:
        return builder.getF64Type();
      default:
        return builder.getF64Type();
    }
  }

  mlir::Value codegen(ExprAST& node) {
    if (auto* block = dynamic_cast<BlockAST*>(&node)) {
      for (auto& expr : block->getExpressions()) {
        codegen(*expr);
      }
      return nullptr;
    }

    if (auto* num = dynamic_cast<NumberExprAST*>(&node)) {
      mlir::Type type = getMLIRType(num->getType());
      if (type.isa<mlir::FloatType>()) {
        return builder.create<mlir::arith::ConstantOp>(
            builder.getUnknownLoc(), builder.getFloatAttr(type, num->getVal()));
      }
      return builder.create<mlir::arith::ConstantOp>(
          builder.getUnknownLoc(),
          builder.getIntegerAttr(type, static_cast<int64_t>(num->getVal())));
    }

    if (auto* varDecl = dynamic_cast<VarDeclAST*>(&node)) {
      auto initVal = codegen(*varDecl->getInitVal());
      if (!initVal) return nullptr;

      mlir::Type targetType;
      if (varDecl->getType().type == DataType::Inferred) {
        targetType = initVal.getType();
      } else {
        targetType = getMLIRType(varDecl->getType().type);
        // TODO(gemini): Handle type casting if types don't match
      }

      symbolTable[varDecl->getName()] = initVal;
      return initVal;
    }

    if (auto* varExpr = dynamic_cast<VariableExprAST*>(&node)) {
      if (symbolTable.count(varExpr->getName())) {
        return symbolTable[varExpr->getName()];
      }
      return nullptr;
    }

    if (auto* bin = dynamic_cast<BinaryExprAST*>(&node)) {
      mlir::Value lhs = codegen(*bin->getLHS());
      mlir::Value rhs = codegen(*bin->getRHS());
      if (!lhs || !rhs) return nullptr;

      bool isFloat = lhs.getType().isa<mlir::FloatType>();

      switch (bin->getOp()) {
        case '+':
          if (isFloat)
            return builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          return builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '-':
          if (isFloat)
            return builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          return builder.create<mlir::arith::SubIOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '*':
          if (isFloat)
            return builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          return builder.create<mlir::arith::MulIOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '/':
          if (isFloat)
            return builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          // TODO(gemini): handle signed/unsigned division
          return builder.create<mlir::arith::DivSIOp>(builder.getUnknownLoc(),
                                                      lhs, rhs);
      }
    }
    return nullptr;
  }

  mlir::MLIRContext& context;
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
  std::map<std::string, mlir::Value> symbolTable;
};

mlir::ModuleOp mlirGen(mlir::MLIRContext& context, ExprAST& ast) {
  return MLIRGenImpl(context).generate(ast);
}

}  // namespace toy
