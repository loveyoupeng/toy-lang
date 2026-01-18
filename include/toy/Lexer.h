#ifndef TOY_LEXER_H_
#define TOY_LEXER_H_

#include <string>
#include <utility>

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
};

class Lexer {
 public:
  explicit Lexer(std::string content)
      : content(std::move(content)), lastChar(' ') {}

  int gettok();
  const std::string& getIdentifier() const {
    return identifierStr;
  }
  double getNumVal() const {
    return numVal;
  }

 private:
  std::string content;
  size_t pos = 0;
  int lastChar;
  std::string identifierStr;
  double numVal;

  int nextChar() {
    if (pos >= content.size()) return EOF;
    return content[pos++];
  }
};

}  // namespace toy

#endif  // TOY_LEXER_H_
