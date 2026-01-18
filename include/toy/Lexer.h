#ifndef TOY_LEXER_H_
#define TOY_LEXER_H_

#include <string>
#include <utility>

#include "toy/AST.h"

namespace toy {

enum Token {
  tok_eof = -1,

  // commands
  tok_var = -2,
  tok_val = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,

  // types
  tok_type_byte = -10,
  tok_type_int8 = -11,
  tok_type_int16 = -12,
  tok_type_int32 = -13,
  tok_type_int64 = -14,
  tok_type_uint8 = -15,
  tok_type_uint32 = -16,
  tok_type_uint64 = -17,
  tok_type_float32 = -18,
  tok_type_float64 = -19,

  // built-in macros
  tok_as_int8 = -30,
  tok_as_int16 = -31,
  tok_as_int32 = -32,
  tok_as_int64 = -33,
  tok_as_uint8 = -34,
  tok_as_uint32 = -35,
  tok_as_uint64 = -36,
  tok_as_float32 = -37,
  tok_as_float64 = -22,
  tok_as_byte = -23,

  // Builtins
  tok_print = -24,
  tok_println = -25,
  tok_stderr = -26,

  // Literals
  tok_string_literal = -27,
};

class Lexer {
  std::string identifierStr;
  std::string stringVal;  // For string literals
  double numVal;
  char lastChar = ' ';
  int curLine = 1;
  int curCol = 0;
  Location lastLoc = {1, 0};
  const std::string source;
  int curPos = 0;

  char nextChar() {
    if (curPos >= source.size()) return EOF;
    char c = source[curPos++];
    if (c == '\n') {
      curLine++;
      curCol = 0;
    } else {
      curCol++;
    }
    return c;
  }

 public:
  explicit Lexer(std::string source) : source(std::move(source)) {}

  std::string getIdentifier() const {
    return identifierStr;
  }
  std::string getStringVal() const {
    return stringVal;
  }
  double getNumVal() const {
    return numVal;
  }
  Location getLastLoc() const {
    return lastLoc;
  }

  int gettok();
};

}  // namespace toy

#endif  // TOY_LEXER_H_
