set(LLVM_LINK_COMPONENTS
  Support
  Core
  Target
  TargetParser
  Analysis
  ExecutionEngine
  OrcJIT
)

add_executable(toy-lang 
  main.cpp
  frontend/Lexer.cpp
  frontend/Parser.cpp
  mlir/MLIRGen.cpp
)

target_include_directories(toy-lang PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(toy-lang PRIVATE
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
