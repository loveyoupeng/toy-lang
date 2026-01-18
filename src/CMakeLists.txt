set(LLVM_LINK_COMPONENTS
  Support
  Core
  Target
  TargetParser
  Analysis
  ExecutionEngine
  OrcJIT
)

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
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

add_executable(toy-lang main.cpp)
target_link_libraries(toy-lang PRIVATE toy_compiler)