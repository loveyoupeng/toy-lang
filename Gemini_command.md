
## Commit: d077d187905a1629bce73bfbe33e4de60811270a
**Date:** Sun Jan 18 19:07:07 2026 +0800
**Author:** loveyoupeng

**Message:**
Initialize project structure and Gemini automation hooks

---

## Commit: 9d167736001685c981e8b939a9612abe2d207da7
**Date:** Sun Jan 18 19:07:44 2026 +0800
**Author:** loveyoupeng

**Message:**
Initialize project structure and Gemini automation hooks

Summary:
1. Created project directories: src, include, tests, benchmarks, cmake.
2. Configured root CMakeLists.txt with CPM.cmake for googletest and benchmark.
3. Added MLIR configuration to CMake.
4. Created basic src/main.cpp and test/benchmark examples.
5. Implemented git pre-commit hook for unit test validation.
6. Implemented scripts/gemini_push.py and scripts/generate_log.py for automated commit logging.

Automated commit by Gemini CLI.

---

## Commit: 930f277e3c8215db236f90b228f813885bc8653f
**Date:** Sun Jan 18 19:17:23 2026 +0800
**Author:** loveyoupeng

**Message:**
Update project structure for a semi-general programming language and implement basic MLIR generation for numeric expressions.

Summary:
1. Reorganized source tree into frontend, mlir, backend, and runtime folders.
2. Created toy::AST namespace and basic expression nodes.
3. Implemented toy::mlirGen to convert AST to MLIR (Arith and Func dialects).
4. Updated main.cpp to demonstrate MLIR generation for '1.0 + 1.0'.
5. Updated CMake configuration to support new MLIR dependencies.

Automated commit by Gemini CLI.

---
