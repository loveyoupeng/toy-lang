#include "toy/Lexer.h"

#include <cctype>
#include <map>
#include <string>

namespace toy {

int Lexer::gettok() {
  while (isspace(lastChar)) lastChar = nextChar();

  lastLoc.line = curLine;
  lastLoc.col = curCol;

  if (isalpha(lastChar)) {
    identifierStr = static_cast<char>(lastChar);
    while (isalnum((lastChar = nextChar())))
      identifierStr += static_cast<char>(lastChar);

    if (identifierStr == "var") return tok_var;
    if (identifierStr == "val") return tok_val;

    // Byte literal prefix: bx
    if (identifierStr == "bx" && isdigit(lastChar)) {
      std::string numStr;
      while (isdigit(lastChar)) {
        numStr += static_cast<char>(lastChar);
        lastChar = nextChar();
      }
      numVal = strtod(numStr.c_str(), nullptr);
      return tok_number;  // Will be handled as Byte by type of literal logic
    }

    static const std::map<std::string, int> keywordMap = {
        {"byte", tok_type_byte},       {"int8", tok_type_int8},
        {"int16", tok_type_int16},     {"int32", tok_type_int32},
        {"int", tok_type_int32},       {"int64", tok_type_int64},
        {"uint8", tok_type_uint8},     {"uint32", tok_type_uint32},
        {"uint64", tok_type_uint64},   {"float32", tok_type_float32},
        {"float64", tok_type_float64}, {"double", tok_type_float64},
        {"asint8", tok_as_int8},       {"asint16", tok_as_int16},
        {"asint32", tok_as_int32},     {"asint64", tok_as_int64},
        {"asuint8", tok_as_uint8},     {"asuint32", tok_as_uint32},
        {"asuint64", tok_as_uint64},   {"asfloat32", tok_as_float32},
        {"asfloat64", tok_as_float64}, {"asbyte", tok_as_byte},
    };

    if (keywordMap.count(identifierStr)) return keywordMap.at(identifierStr);

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
