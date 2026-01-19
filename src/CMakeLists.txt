set(LLVM_LINK_COMPONENTS
  Support
  Core
  Target
  TargetParser
  Analysis
  ExecutionEngine
  OrcJIT
  X86CodeGen
  X86AsmParser
  X86Desc
  X86Info
)

llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})

add_library(toy_compiler
  frontend/Lexer.cpp
  frontend/Parser.cpp
  mlir/MLIRGen.cpp
)

target_include_directories(toy_compiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(toy_compiler PUBLIC
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRPass
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  ${llvm_libs}
)

add_executable(toy-lang main.cpp)
target_link_libraries(toy-lang PRIVATE toy_compiler ${llvm_libs})