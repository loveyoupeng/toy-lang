#include "toy/MLIRGen.h"

#include <iostream>
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

  // Helper to check if a type is an integer (including Byte)
  bool isInteger(mlir::Type t) {
    return t.isa<mlir::IntegerType>();
  }
  bool isFloat(mlir::Type t) {
    return t.isa<mlir::FloatType>();
  }
  bool isByte(mlir::Type t) {
    auto it = t.dyn_cast<mlir::IntegerType>();
    return it && it.getWidth() == 8 && !it.isSigned();
  }

  int getWidth(mlir::Type t) {
    if (auto it = t.dyn_cast<mlir::IntegerType>()) return it.getWidth();
    if (t.isF32()) return 32;
    if (t.isF64()) return 64;
    return 0;
  }

  mlir::Value emitCast(mlir::Value value, mlir::Type destType, Location loc,
                       bool isExplicit = false) {
    mlir::Type srcType = value.getType();
    if (srcType == destType) return value;

    // Byte restrictions
    if (!isExplicit) {
      if (isByte(srcType) || isByte(destType)) {
        std::cerr << "Error [" << loc.line << ":" << loc.col
                  << "]: Implicit conversion involving 'byte' is not allowed. "
                     "Use explicit asType() macro.\n";
        return nullptr;
      }
    }

    // Integer to Integer promotion
    if (isInteger(srcType) && isInteger(destType)) {
      if (getWidth(destType) > getWidth(srcType)) {
        // Promotion (Sign or Zero extend)
        if (destType.isUnsignedInteger())
          return builder.create<mlir::arith::ExtUIOp>(builder.getUnknownLoc(),
                                                      destType, value);
        return builder.create<mlir::arith::ExtSIOp>(builder.getUnknownLoc(),
                                                    destType, value);
      } else if (getWidth(destType) < getWidth(srcType)) {
        if (!isExplicit) {
          std::cerr << "Error [" << loc.line << ":" << loc.col
                    << "]: Illegal implicit truncation from "
                    << getWidth(srcType) << "-bit to " << getWidth(destType)
                    << "-bit.\n";
          return nullptr;
        }
        return builder.create<mlir::arith::TruncIOp>(builder.getUnknownLoc(),
                                                     destType, value);
      }
    }

    // Float to Float promotion
    if (isFloat(srcType) && isFloat(destType)) {
      if (getWidth(destType) > getWidth(srcType)) {
        return builder.create<mlir::arith::ExtFOp>(builder.getUnknownLoc(),
                                                   destType, value);
      } else if (getWidth(destType) < getWidth(srcType)) {
        if (!isExplicit) {
          std::cerr << "Error [" << loc.line << ":" << loc.col
                    << "]: Illegal implicit float truncation.\n";
          return nullptr;
        }
        return builder.create<mlir::arith::TruncFOp>(builder.getUnknownLoc(),
                                                     destType, value);
      }
    }

    // Explicit casts between int and float
    if (isExplicit) {
      if (isInteger(srcType) && isFloat(destType)) {
        if (srcType.isUnsignedInteger())
          return builder.create<mlir::arith::UIToFPOp>(builder.getUnknownLoc(),
                                                       destType, value);
        return builder.create<mlir::arith::SIToFPOp>(builder.getUnknownLoc(),
                                                     destType, value);
      }
      if (isFloat(srcType) && isInteger(destType)) {
        if (destType.isUnsignedInteger())
          return builder.create<mlir::arith::FPToUIOp>(builder.getUnknownLoc(),
                                                       destType, value);
        return builder.create<mlir::arith::FPToSIOp>(builder.getUnknownLoc(),
                                                     destType, value);
      }
      // Bitcast for byte/int8 if same width
      if (getWidth(srcType) == getWidth(destType)) {
        return value;  // MLIR often treats i8 and ui8 similarly in some ops,
                       // but we can add bitcast if needed.
      }
    }

    std::cerr << "Error [" << loc.line << ":" << loc.col
              << "]: Unsupported conversion.\n";
    return nullptr;
  }

  mlir::Value codegen(ExprAST& node) {
    if (auto* block = dynamic_cast<BlockAST*>(&node)) {
      for (auto& expr : block->getExpressions()) {
        if (!codegen(*expr)) return nullptr;
      }
      return builder.getUnknownLoc()
          .getContext()
          ->getOrLoadDialect<mlir::arith::ArithDialect>()
          ->getNamespace();  // Dummy
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
        initVal = emitCast(initVal, targetType, varDecl->loc());
        if (!initVal) return nullptr;
      }

      symbolTable[varDecl->getName()] = initVal;
      return initVal;
    }

    if (auto* varExpr = dynamic_cast<VariableExprAST*>(&node)) {
      if (symbolTable.count(varExpr->getName())) {
        return symbolTable[varExpr->getName()];
      }
      std::cerr << "Error [" << varExpr->loc().line << ":" << varExpr->loc().col
                << "]: Unknown variable '" << varExpr->getName() << "'\n";
      return nullptr;
    }

    if (auto* cast = dynamic_cast<CastExprAST*>(&node)) {
      auto arg = codegen(*cast->getArg());
      if (!arg) return nullptr;
      return emitCast(arg, getMLIRType(cast->getDestType()), cast->loc(), true);
    }

    if (auto* bin = dynamic_cast<BinaryExprAST*>(&node)) {
      mlir::Value lhs = codegen(*bin->getLHS());
      mlir::Value rhs = codegen(*bin->getRHS());
      if (!lhs || !rhs) return nullptr;

      // Type Promotion
      mlir::Type lType = lhs.getType();
      mlir::Type rType = rhs.getType();

      if (lType != rType) {
        if (isByte(lType) || isByte(rType)) {
          std::cerr << "Error [" << bin->loc().line << ":" << bin->loc().col
                    << "]: Operations with 'byte' must use explicit "
                       "conversion.\n";
          return nullptr;
        }

        // Promote smaller to larger
        if (getWidth(lType) < getWidth(rType)) {
          lhs = emitCast(lhs, rType, bin->loc());
          if (!lhs) return nullptr;
        } else if (getWidth(rType) < getWidth(lType)) {
          rhs = emitCast(rhs, lType, bin->loc());
          if (!rhs) return nullptr;
        } else {
          // Same width, different types (e.g. i32 vs f32) - requires explicit
          std::cerr << "Error [" << bin->loc().line << ":" << bin->loc().col
                    << "]: Implicit conversion between different types of "
                       "same width not supported.\n";
          return nullptr;
        }
      }

      mlir::Type resType = lhs.getType();
      bool isF = resType.isa<mlir::FloatType>();

      switch (bin->getOp()) {
        case '+':
          if (isF)
            return builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          return builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '-':
          if (isF)
            return builder.create<mlir::arith::SubFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          return builder.create<mlir::arith::SubIOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '*':
          if (isF)
            return builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          return builder.create<mlir::arith::MulIOp>(builder.getUnknownLoc(),
                                                     lhs, rhs);
        case '/':
          if (isF)
            return builder.create<mlir::arith::DivFOp>(builder.getUnknownLoc(),
                                                       lhs, rhs);
          if (resType.isUnsignedInteger())
            return builder.create<mlir::arith::DivUIOp>(builder.getUnknownLoc(),
                                                        lhs, rhs);
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
