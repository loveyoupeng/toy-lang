#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

int main(int argc, char **argv) {
    mlir::MLIRContext context;
    context.loadAllAvailableDialects();

    llvm::outs() << "Toy Lang MLIR Compiler Initialized\n";
    return 0;
}

