#include "toy/Lexer.h"

#include <cctype>
#include <map>
#include <string>

namespace toy {

int Lexer::gettok() {
  while (isspace(lastChar)) lastChar = nextChar();

  if (isalpha(lastChar)) {
    identifierStr = static_cast<char>(lastChar);
    while (isalnum((lastChar = nextChar())))
      identifierStr += static_cast<char>(lastChar);

    if (identifierStr == "var") return tok_var;
    if (identifierStr == "val") return tok_val;

    static const std::map<std::string, int> typeMap = {
        {"byte", tok_type_byte},       {"int8", tok_type_int8},
        {"int16", tok_type_int16},     {"int32", tok_type_int32},
        {"int", tok_type_int32},       {"int64", tok_type_int64},
        {"uint8", tok_type_uint8},     {"uint32", tok_type_uint32},
        {"uint64", tok_type_uint64},   {"float32", tok_type_float32},
        {"float64", tok_type_float64}, {"double", tok_type_float64},
    };

    if (typeMap.count(identifierStr)) return typeMap.at(identifierStr);

    return tok_identifier;
  }

  if (isdigit(lastChar) || lastChar == '.') {
    std::string numStr;
    do {
      numStr += static_cast<char>(lastChar);
      lastChar = nextChar();
    } while (isdigit(lastChar) || lastChar == '.');

    numVal = strtod(numStr.c_str(), nullptr);
    return tok_number;
  }

  if (lastChar == EOF) return tok_eof;

  int thisChar = lastChar;
  lastChar = nextChar();
  return thisChar;
}

}  // namespace toy
