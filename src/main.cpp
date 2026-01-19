#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "toy/AST.h"
#include "toy/Lexer.h"
#include "toy/MLIRGen.h"
#include "toy/Parser.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: toy-lang <filename.toy> [-o <output>]\n";
    return 1;
  }

  std::string filename = argv[1];
  std::string outFilename;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "-o" && i + 1 < argc) {
      outFilename = argv[++i];
    }
  }

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
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  toy::Lexer lexer(content);
  toy::Parser parser(std::move(lexer));
  auto ast = parser.parse();

  if (!ast) {
    std::cerr << "Parsing failed at " << filename << "\n";
    return 1;
  }

  auto module = toy::mlirGen(context, *ast);

  if (!module) {
    llvm::errs() << "Failed to generate MLIR\n";
    return 1;
  }

  if (outFilename.empty()) {
    module.print(llvm::outs());
    return 0;
  }

  // --- Lowering ---
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(module.getOperation()))) {
    llvm::errs() << "Failed to lower MLIR\n";
    return 1;
  }

  // --- Translation to LLVM IR ---
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  llvm::LLVMContext llvmContext;
  auto llvmModule =
      mlir::translateModuleToLLVMIR(module.getOperation(), llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR\n";
    return 1;
  }

  // --- Code Generation ---
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  llvm::Triple triple(targetTriple);
  llvmModule->setTargetTriple(triple);

  std::string error;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << error;
    return 1;
  }

  auto CPU = "generic";
  auto features = "";
  llvm::TargetOptions opt;
  auto RM = std::optional<llvm::Reloc::Model>();
  auto targetMachine =
      target->createTargetMachine(triple, CPU, features, opt, RM);

  llvmModule->setDataLayout(targetMachine->createDataLayout());

  std::string objFilename = outFilename + ".obj";
  std::error_code EC;
  llvm::raw_fd_ostream dest(objFilename, EC, llvm::sys::fs::OF_None);

  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    return 1;
  }

  llvm::legacy::PassManager pass;
  auto fileType = llvm::CodeGenFileType::ObjectFile;

  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
    llvm::errs() << "TargetMachine can't emit a file of this type";
    return 1;
  }

  pass.run(*llvmModule);
  dest.flush();
  dest.close();

  // --- Linking ---
  auto clangPath = llvm::sys::findProgramByName("clang");
  if (!clangPath) {
    llvm::errs() << "clang not found in PATH\n";
    return 1;
  }

  std::vector<llvm::StringRef> args;
  args.push_back(*clangPath);
  args.push_back(objFilename);
  args.push_back("-o");
  args.push_back(outFilename);

  if (llvm::sys::ExecuteAndWait(*clangPath, args) != 0) {
    llvm::errs() << "Linking failed\n";
    return 1;
  }

  // Cleanup object file
  llvm::sys::fs::remove(objFilename);

  return 0;
}
