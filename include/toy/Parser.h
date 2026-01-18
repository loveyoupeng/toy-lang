#ifndef TOY_PARSER_H_
#define TOY_PARSER_H_

#include <memory>
#include <utility>
#include <vector>

#include "toy/AST.h"
#include "toy/Lexer.h"

namespace toy {

class Parser {
 public:
  explicit Parser(Lexer lexer) : lexer(std::move(lexer)) {
    getNextToken();
  }

  std::unique_ptr<BlockAST> parse();

 private:
  Lexer lexer;
  int curTok;

  int getNextToken() {
    return curTok = lexer.gettok();
  }

  std::unique_ptr<ExprAST> parseExpression();
  std::unique_ptr<ExprAST> parsePrimary();
  std::unique_ptr<ExprAST> parseBinOpRHS(Location loc, int exprPrec,
                                         std::unique_ptr<ExprAST> lhs);
  std::unique_ptr<ExprAST> parseNumberExpr();
  std::unique_ptr<ExprAST> parseIdentifierExpr();
  std::unique_ptr<ExprAST> parseVarDecl(bool isConstant);
  std::unique_ptr<ExprAST> parsePrintExpr();
  DataType parseType();
};

}  // namespace toy

#endif  // TOY_PARSER_H_
