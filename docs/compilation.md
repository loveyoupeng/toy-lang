# How to Compile Toy Language to Executable

This guide describes how to compile a `.toy` source file into a standalone executable (e.g., `.exe` on Windows or binary on Linux) using the Toy compiler and LLVM tools.

## Prerequisites

1.  **Toy Compiler (`toy-lang`)**: Built from this repository.
2.  **LLVM Tools**: The following tools must be available (built from LLVM source with MLIR enabled):
    *   `mlir-opt`
    *   `mlir-translate`
    *   `clang` (or system compiler capable of linking LLVM IR)

## Compilation Pipeline

The compilation process involves lowering the Toy AST to MLIR, refining the MLIR to the LLVM dialect, translating it to LLVM IR, and finally compiling/linking it to machine code.

### Step 1: Generate MLIR from Toy Source

Use the `toy-lang` compiler to parse the `.toy` file and generate the initial MLIR (Toy dialect, Arith dialect, etc.).

```bash
# Windows (PowerShell/CMD) - use cmd /c to ensure correct encoding/redirection if needed
cmd /c "build\src\toy-lang.exe test.toy > test.mlir"

# Linux/Mac
./build/src/toy-lang test.toy > test.mlir
```

### Step 2: Lower MLIR to LLVM Dialect

Use `mlir-opt` to run conversion passes that transform high-level dialects (Arith, Func) into the LLVM dialect.

```bash
path/to/mlir-opt \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  test.mlir -o test_lowered.mlir
```

### Step 3: Translate to LLVM IR

Use `mlir-translate` to convert the LLVM dialect MLIR into standard LLVM IR (`.ll` file).

```bash
path/to/mlir-translate \
  --mlir-to-llvmir \
  test_lowered.mlir -o test.ll
```

### Step 4: Compile to Executable

Use `clang` to compile the LLVM IR and link it into an executable.

```bash
clang test.ll -o test.exe
```

### Step 5: Run

```bash
./test.exe
```

## Example (Windows)

Assuming `toy-lang` is in `build\src` and LLVM tools are in `D:\Projects\opensource\llvm-project\build\bin`:

```powershell
# 1. Gen MLIR
cmd /c "build\src\toy-lang.exe test_print.toy > test_print.mlir"

# 2. Lower to LLVM Dialect
D:\Projects\opensource\llvm-project\build\bin\mlir-opt.exe \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
  test_print.mlir -o test_print_lowered.mlir

# 3. Translate to .ll
D:\Projects\opensource\llvm-project\build\bin\mlir-translate.exe \
  --mlir-to-llvmir \
  test_print_lowered.mlir -o test_print.ll

# 4. Compile
& "D:\Program Files\LLVM\bin\clang.exe" test_print.ll -o test_print.exe

# 5. Run
.\test_print.exe
```
