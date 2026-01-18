#include "toy/MLIRGen.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

    if (!codegen(ast)) return nullptr;

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
    return llvm::isa<mlir::IntegerType>(t);
  }
  bool isFloat(mlir::Type t) {
    return llvm::isa<mlir::FloatType>(t);
  }
  bool isByte(mlir::Type t) {
    auto it = llvm::dyn_cast<mlir::IntegerType>(t);
    return it && it.getWidth() == 8 && !it.isSigned();
  }

  int getWidth(mlir::Type t) {
    if (auto it = llvm::dyn_cast<mlir::IntegerType>(t)) return it.getWidth();
    if (t.isF32()) return 32;
    if (t.isF64()) return 64;
    return 0;
  }

  mlir::Location getLoc(Location loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr("toy_source"),
                                     loc.line, loc.col);
  }

  mlir::Value getOrCreateGlobalString(Location loc, llvm::StringRef msg) {
    static int strCount = 0;

    std::string name = "str_" + std::to_string(strCount++);

    auto type = mlir::LLVM::LLVMArrayType::get(builder.getIntegerType(8),
                                               msg.size() + 1);

    mlir::LLVM::GlobalOp global;

    {
      mlir::OpBuilder::InsertionGuard guard(builder);

      builder.setInsertionPointToStart(module.getBody());

      std::string val = msg.str();

      val.push_back('\0');

      global = builder.create<mlir::LLVM::GlobalOp>(

          builder.getUnknownLoc(),

          type,

          /*isConstant=*/true,

          mlir::LLVM::Linkage::Internal,

          name,

          builder.getStringAttr(val));
    }

    auto addr = builder.create<mlir::LLVM::AddressOfOp>(getLoc(loc), global);

    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(

        getLoc(loc), builder.getIntegerType(64),
        builder.getIntegerAttr(builder.getIntegerType(64), 0));

    std::vector<mlir::Value> indices = {zero, zero};

    return builder.create<mlir::LLVM::GEPOp>(

        getLoc(loc), mlir::LLVM::LLVMPointerType::get(builder.getContext()),

        global.getType(), addr, indices);
  }

  mlir::LLVM::LLVMFuncOp getPrintf() {
    auto lookup = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    if (lookup) return lookup;

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto llvmI32 = builder.getIntegerType(32);
    auto llvmPtr = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32, {llvmPtr},
                                                        /*isVarArg=*/true);

    return builder.create<mlir::LLVM::LLVMFuncOp>(builder.getUnknownLoc(),
                                                  "printf", llvmFnType);
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

    // Implicit Int to Float promotion
    if (!isExplicit && isInteger(srcType) && isFloat(destType)) {
      if (srcType.isUnsignedInteger())
        return builder.create<mlir::arith::UIToFPOp>(builder.getUnknownLoc(),
                                                     destType, value);
      return builder.create<mlir::arith::SIToFPOp>(builder.getUnknownLoc(),
                                                   destType, value);
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
      // Return a dummy value (e.g. 0.0) as block result for now.
      return builder.create<mlir::arith::ConstantOp>(
          builder.getUnknownLoc(), builder.getF64FloatAttr(0.0));
    }

    if (auto* num = dynamic_cast<NumberExprAST*>(&node)) {
      mlir::Type type = getMLIRType(num->getType());
      if (llvm::isa<mlir::FloatType>(type)) {
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

    if (auto* strExpr = dynamic_cast<StringExprAST*>(&node)) {
      return getOrCreateGlobalString(strExpr->loc(), strExpr->getValue());
    }

    if (auto* printExpr = dynamic_cast<PrintExprAST*>(&node)) {
      auto printfFunc = getPrintf();
      Location loc = printExpr->loc();
      auto mlirLoc = getLoc(loc);

      for (auto& arg : printExpr->getArgs()) {
        if (auto* strArg = dynamic_cast<StringExprAST*>(arg.get())) {
          auto val = codegen(*arg);
          builder.create<mlir::LLVM::CallOp>(mlirLoc, printfFunc, val);
        } else {
          auto val = codegen(*arg);
          if (!val) return nullptr;

          std::string fmt;
          if (llvm::isa<mlir::FloatType>(val.getType()))
            fmt = "%f";
          else
            fmt = "%d";

          auto fmtStr = getOrCreateGlobalString(loc, fmt);
          builder.create<mlir::LLVM::CallOp>(
              mlirLoc, printfFunc, std::vector<mlir::Value>{fmtStr, val});
        }
      }

      if (printExpr->getIsNewLine()) {
        auto nlStr = getOrCreateGlobalString(loc, "\n");
        builder.create<mlir::LLVM::CallOp>(mlirLoc, printfFunc, nlStr);
      }

      return builder.create<mlir::arith::ConstantIntOp>(mlirLoc, 0, 32);
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
      bool isF = llvm::isa<mlir::FloatType>(resType);

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
