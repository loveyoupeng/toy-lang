set(LLVM_LINK_COMPONENTS
  Support
  Core
)

add_executable(toy-lang main.cpp)

target_link_libraries(toy-lang PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
)
