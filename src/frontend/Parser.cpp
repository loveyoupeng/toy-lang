#include "toy/Parser.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace toy {

static std::map<char, int> BinopPrecedence = {
    {'+', 20}, {'-', 20}, {'*', 40}, {'/', 40}};

int getTokPrecedence(int tok) {
  if (!isascii(tok)) return -1;
  int prec = BinopPrecedence[tok];
  if (prec <= 0) return -1;
  return prec;
}

std::unique_ptr<BlockAST> Parser::parse() {
  std::vector<std::unique_ptr<ExprAST>> expressions;
  while (curTok != tok_eof) {
    if (curTok == ';') {
      getNextToken();
    } else if (curTok == tok_var) {
      expressions.push_back(parseVarDecl(false));
    } else if (curTok == tok_val) {
      expressions.push_back(parseVarDecl(true));
    } else {
      expressions.push_back(parseExpression());
    }
  }
  return std::make_unique<BlockAST>(std::move(expressions));
}

std::unique_ptr<ExprAST> Parser::parseVarDecl(bool isConstant) {
  getNextToken();  // eat var/val

  if (curTok != tok_identifier) return nullptr;
  std::string name = lexer.getIdentifier();
  getNextToken();

  DataType type = DataType::Inferred;
  if (curTok == ':') {
    getNextToken();
    type = parseType();
  }

  if (curTok != '=') return nullptr;
  getNextToken();

  auto initVal = parseExpression();

  if (curTok == ';') getNextToken();

  return std::make_unique<VarDeclAST>(name, VarType{type}, std::move(initVal),
                                      isConstant);
}

DataType Parser::parseType() {
  DataType type = DataType::Inferred;
  switch (curTok) {
    case tok_type_byte:
      type = DataType::Byte;
      break;
    case tok_type_int8:
      type = DataType::Int8;
      break;
    case tok_type_int16:
      type = DataType::Int16;
      break;
    case tok_type_int32:
      type = DataType::Int32;
      break;
    case tok_type_int64:
      type = DataType::Int64;
      break;
    case tok_type_uint8:
      type = DataType::UInt8;
      break;
    case tok_type_uint32:
      type = DataType::UInt32;
      break;
    case tok_type_uint64:
      type = DataType::UInt64;
      break;
    case tok_type_float32:
      type = DataType::Float32;
      break;
    case tok_type_float64:
      type = DataType::Float64;
      break;
  }
  getNextToken();
  return type;
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
  auto lhs = parsePrimary();
  if (!lhs) return nullptr;
  return parseBinOpRHS(0, std::move(lhs));
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
  switch (curTok) {
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_number:
      return parseNumberExpr();
    case '(': {
      getNextToken();
      auto v = parseExpression();
      if (curTok != ')') return nullptr;
      getNextToken();
      return v;
    }
    default:
      return nullptr;
  }
}

std::unique_ptr<ExprAST> Parser::parseNumberExpr() {
  auto result = std::make_unique<NumberExprAST>(lexer.getNumVal());
  getNextToken();
  return result;
}

std::unique_ptr<ExprAST> Parser::parseIdentifierExpr() {
  std::string name = lexer.getIdentifier();
  getNextToken();
  return std::make_unique<VariableExprAST>(name);
}

std::unique_ptr<ExprAST> Parser::parseBinOpRHS(int exprPrec,
                                               std::unique_ptr<ExprAST> lhs) {
  while (true) {
    int tokPrec = getTokPrecedence(curTok);
    if (tokPrec < exprPrec) return lhs;

    int binOp = curTok;
    getNextToken();

    auto rhs = parsePrimary();
    if (!rhs) return nullptr;

    int nextPrec = getTokPrecedence(curTok);
    if (tokPrec < nextPrec) {
      rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
      if (!rhs) return nullptr;
    }

    lhs =
        std::make_unique<BinaryExprAST>(binOp, std::move(lhs), std::move(rhs));
  }
}

}  // namespace toy
