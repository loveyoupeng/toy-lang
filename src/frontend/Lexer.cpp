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
    if (identifierStr == "if") return tok_if;
    if (identifierStr == "else") return tok_else;
    if (identifierStr == "true") return tok_true;
    if (identifierStr == "false") return tok_false;

    // Byte literal prefix: bx<digits>
    if (identifierStr.size() > 2 && identifierStr.substr(0, 2) == "bx") {
      bool isByteLiteral = true;
      for (size_t i = 2; i < identifierStr.size(); ++i) {
        if (!isdigit(identifierStr[i])) {
          isByteLiteral = false;
          break;
        }
      }
      if (isByteLiteral) {
        std::string numStr = identifierStr.substr(2);
        numVal = strtod(numStr.c_str(), nullptr);
        return tok_number;
      }
    }

    static const std::map<std::string, int> keywordMap = {
        {"byte", tok_type_byte},       {"bool", tok_type_bool},
        {"int8", tok_type_int8},       {"int16", tok_type_int16},

        {"int32", tok_type_int32},     {"int", tok_type_int32},
        {"int64", tok_type_int64},     {"uint8", tok_type_uint8},
        {"uint32", tok_type_uint32},   {"uint64", tok_type_uint64},
        {"float32", tok_type_float32}, {"float64", tok_type_float64},
        {"double", tok_type_float64},  {"asint8", tok_as_int8},
        {"asint16", tok_as_int16},     {"asint32", tok_as_int32},
        {"asint64", tok_as_int64},     {"asuint8", tok_as_uint8},
        {"asuint32", tok_as_uint32},   {"asuint64", tok_as_uint64},
        {"asfloat32", tok_as_float32}, {"asfloat64", tok_as_float64},
        {"asbyte", tok_as_byte},       {"print", tok_print},
        {"println", tok_println},      {"stderr", tok_stderr},
    };

    if (keywordMap.count(identifierStr)) return keywordMap.at(identifierStr);

    return tok_identifier;
  }

  if (lastChar == '"') {
    std::string str;
    lastChar = nextChar();  // eat opening quote
    while (lastChar != '"' && lastChar != EOF) {
      if (lastChar == '\\') {
        lastChar = nextChar();  // skip backslash
        if (lastChar == 'n')
          str += '\n';
        else if (lastChar == 't')
          str += '\t';
        else
          str += static_cast<char>(lastChar);
      } else {
        str += static_cast<char>(lastChar);
      }
      lastChar = nextChar();
    }
    lastChar = nextChar();  // eat closing quote
    stringVal = str;
    return tok_string_literal;
  }

  if (isdigit(lastChar) || lastChar == '.') {
    std::string numStr;
    do {
      numStr += static_cast<char>(lastChar);
      lastChar = nextChar();
    } while (isdigit(lastChar) || lastChar == '.');

    numVal = strtod(numStr.c_str(), nullptr);
    identifierStr = numStr;
    return tok_number;
  }

  if (lastChar == EOF) return tok_eof;

  if (lastChar == '/') {
    lastChar = nextChar();
    if (lastChar == '/') {
      // Single-line comment
      do {
        lastChar = nextChar();
      } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF) return gettok();
    } else {
      return '/';
    }
  }

  int thisChar = lastChar;
  lastChar = nextChar();
  return thisChar;
}

}  // namespace toy
