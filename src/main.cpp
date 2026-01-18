#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "toy/AST.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: toy-lang <filename.toy>\n";
    return 1;
  }

  std::string filename = argv[1];
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open file: " << filename << "\n";
    return 1;
  }

  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string content = ss.str();

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  toy::Lexer lexer(content);
  toy::Parser parser(std::move(lexer));
  auto ast = parser.parse();

  if (!ast) {
    std::cerr << "Parsing failed\n";
    return 1;
  }

  auto module = toy::mlirGen(context, *ast);

  if (!module) {
    llvm::errs() << "Failed to generate MLIR\n";
    return 1;
  }

  module->print(llvm::outs());

  return 0;
}
