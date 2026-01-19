#include "toy/Parser.h"

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
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
  Location loc = lexer.getLastLoc();
  std::vector<std::unique_ptr<ExprAST>> expressions;
  while (curTok != tok_eof) {
    if (curTok == ';') {
      getNextToken();
    } else if (curTok == tok_var) {
      auto decl = parseVarDecl(false);
      if (decl)
        expressions.push_back(std::move(decl));
      else
        getNextToken();  // ensure progress
    } else if (curTok == tok_val) {
      auto decl = parseVarDecl(true);
      if (decl)
        expressions.push_back(std::move(decl));
      else
        getNextToken();  // ensure progress
    } else if (curTok == tok_if) {
      auto ifExpr = parseIfExpr();
      if (ifExpr)
        expressions.push_back(std::move(ifExpr));
      else
        getNextToken();  // ensure progress
    } else {
      auto expr = parseExpression();
      if (expr)
        expressions.push_back(std::move(expr));
      else
        getNextToken();  // ensure progress
    }
  }
  return std::make_unique<BlockAST>(loc, std::move(expressions));
}

std::unique_ptr<BlockAST> Parser::parseBlock() {
  Location loc = lexer.getLastLoc();
  if (curTok != '{') return nullptr;
  getNextToken();  // eat {

  std::vector<std::unique_ptr<ExprAST>> expressions;
  while (curTok != '}' && curTok != tok_eof) {
    if (curTok == ';') {
      getNextToken();
    } else if (curTok == tok_var) {
      auto decl = parseVarDecl(false);
      if (decl)
        expressions.push_back(std::move(decl));
      else
        getNextToken();
    } else if (curTok == tok_val) {
      auto decl = parseVarDecl(true);
      if (decl)
        expressions.push_back(std::move(decl));
      else
        getNextToken();
    } else if (curTok == tok_if) {
      auto ifExpr = parseIfExpr();
      if (ifExpr)
        expressions.push_back(std::move(ifExpr));
      else
        getNextToken();
    } else {
      auto expr = parseExpression();
      if (expr)
        expressions.push_back(std::move(expr));
      else
        getNextToken();
    }
  }

  if (curTok != '}') return nullptr;
  getNextToken();  // eat }

  return std::make_unique<BlockAST>(loc, std::move(expressions));
}

std::unique_ptr<ExprAST> Parser::parseIfExpr() {
  Location loc = lexer.getLastLoc();
  getNextToken();  // eat if

  if (curTok != '(') return nullptr;
  getNextToken();  // eat (

  auto cond = parseExpression();
  if (!cond) return nullptr;

  if (curTok != ')') return nullptr;
  getNextToken();  // eat )

  auto thenBlock = parseBlock();
  if (!thenBlock) return nullptr;

  std::unique_ptr<BlockAST> elseBlock = nullptr;
  if (curTok == tok_else) {
    getNextToken();  // eat else
    if (curTok == tok_if) {
      // else if
      Location elseLoc = lexer.getLastLoc();
      auto ifExpr = parseIfExpr();
      std::vector<std::unique_ptr<ExprAST>> exprs;
      exprs.push_back(std::move(ifExpr));
      elseBlock = std::make_unique<BlockAST>(elseLoc, std::move(exprs));
    } else {
      elseBlock = parseBlock();
    }
  }

  return std::make_unique<IfExprAST>(loc, std::move(cond), std::move(thenBlock),
                                     std::move(elseBlock));
}

std::unique_ptr<ExprAST> Parser::parseVarDecl(bool isConstant) {
  Location loc = lexer.getLastLoc();
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

  return std::make_unique<VarDeclAST>(loc, name, VarType{type},
                                      std::move(initVal), isConstant);
}

std::unique_ptr<ExprAST> Parser::parsePrintExpr() {
  Location loc = lexer.getLastLoc();
  bool isNewLine = (curTok == tok_println);
  getNextToken();  // eat print/println

  if (curTok != '(') return nullptr;
  getNextToken();  // eat (

  bool toStderr = false;
  if (curTok == tok_stderr) {
    toStderr = true;
    getNextToken();
    if (curTok == ',') {
      getNextToken();
    } else {
      // If just print(stderr), we might expect a closing ) or comma.
      // But usually print(stderr, "msg").
    }
  }

  if (curTok != tok_string_literal) {
    // Only support string literal format for now
    return nullptr;
  }
  std::string fmt = lexer.getStringVal();
  getNextToken();  // eat string literal

  if (curTok != ')') return nullptr;
  getNextToken();  // eat )

  // Parse interpolation
  std::vector<std::unique_ptr<ExprAST>> args;
  std::string currentStr;
  for (size_t i = 0; i < fmt.size(); ++i) {
    if (fmt[i] == '{') {
      if (!currentStr.empty()) {
        args.push_back(std::make_unique<StringExprAST>(loc, currentStr));
        currentStr.clear();
      }
      // Extract expression string
      std::string exprStr;
      i++;
      int balance = 1;
      while (i < fmt.size() && balance > 0) {
        if (fmt[i] == '{') balance++;
        if (fmt[i] == '}') balance--;
        if (balance > 0) exprStr += fmt[i];
        i++;
      }
      i--;  // adjust for loop increment

      if (balance == 0 && !exprStr.empty()) {
        // Parse the expression
        Lexer subLexer(exprStr);
        Parser subParser(std::move(subLexer));
        auto expr = subParser.parseExpression();
        if (expr) {
          args.push_back(std::move(expr));
        }
      }
    } else {
      currentStr += fmt[i];
    }
  }
  if (!currentStr.empty()) {
    args.push_back(std::make_unique<StringExprAST>(loc, currentStr));
  }

  return std::make_unique<PrintExprAST>(loc, isNewLine, toStderr,
                                        std::move(args));
}

DataType Parser::parseType() {
  DataType type = DataType::Inferred;
  switch (curTok) {
    case tok_type_bool:
      type = DataType::Bool;
      break;
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
  Location loc = lexer.getLastLoc();
  auto lhs = parsePrimary();
  if (!lhs) {
    std::cerr << "Error [" << loc.line << ":" << loc.col
              << "]: Expected primary expression, got token " << curTok << "\n";
    return nullptr;
  }

  // Check for assignment
  if (curTok == '=') {
    getNextToken();  // eat =
    auto rhs = parseExpression();
    if (!rhs) return nullptr;

    auto* varLHS = dynamic_cast<VariableExprAST*>(lhs.get());
    if (!varLHS) {
      std::cerr << "Error [" << loc.line << ":" << loc.col
                << "]: Left-hand side of assignment must be a variable.\n";
      return nullptr;
    }

    return std::make_unique<BinaryExprAST>(loc, '=', std::move(lhs),
                                           std::move(rhs));
  }

  return parseBinOpRHS(loc, 0, std::move(lhs));
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
  Location loc = lexer.getLastLoc();
  switch (curTok) {
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_number:
      return parseNumberExpr();
    case tok_string_literal: {
      std::string val = lexer.getStringVal();
      getNextToken();
      return std::make_unique<StringExprAST>(loc, val);
    }
    case tok_true:
      getNextToken();
      return std::make_unique<BoolExprAST>(loc, true);
    case tok_false:
      getNextToken();
      return std::make_unique<BoolExprAST>(loc, false);
    case tok_if:
      return parseIfExpr();
    case tok_print:
    case tok_println:
      return parsePrintExpr();
    case '(': {
      getNextToken();
      auto v = parseExpression();
      if (curTok != ')') return nullptr;
      getNextToken();
      return v;
    }
    case tok_as_int8:
    case tok_as_int16:
    case tok_as_int32:
    case tok_as_int64:
    case tok_as_uint8:
    case tok_as_uint32:
    case tok_as_uint64:
    case tok_as_float32:
    case tok_as_float64:
    case tok_as_byte: {
      int macroTok = curTok;
      getNextToken();
      if (curTok != '(') return nullptr;
      getNextToken();
      auto arg = parseExpression();
      if (curTok != ')') return nullptr;
      getNextToken();

      DataType destType;
      switch (macroTok) {
        case tok_as_int8:
          destType = DataType::Int8;
          break;
        case tok_as_int16:
          destType = DataType::Int16;
          break;
        case tok_as_int32:
          destType = DataType::Int32;
          break;
        case tok_as_int64:
          destType = DataType::Int64;
          break;
        case tok_as_uint8:
          destType = DataType::UInt8;
          break;
        case tok_as_uint32:
          destType = DataType::UInt32;
          break;
        case tok_as_uint64:
          destType = DataType::UInt64;
          break;
        case tok_as_float32:
          destType = DataType::Float32;
          break;
        case tok_as_float64:
          destType = DataType::Float64;
          break;
        case tok_as_byte:
          destType = DataType::Byte;
          break;
        default:
          return nullptr;
      }
      return std::make_unique<CastExprAST>(loc, destType, std::move(arg));
    }
    default:
      return nullptr;
  }
}

std::unique_ptr<ExprAST> Parser::parseNumberExpr() {
  Location loc = lexer.getLastLoc();
  DataType type = DataType::Float64;

  std::string id = lexer.getIdentifier();
  if (id.size() >= 2 && id.substr(0, 2) == "bx") {
    type = DataType::Byte;
  } else {
    if (id.find('.') != std::string::npos) {
      type = DataType::Float64;
    } else {
      type = DataType::Int32;
    }
  }

  auto result = std::make_unique<NumberExprAST>(loc, lexer.getNumVal(), type);
  getNextToken();
  return result;
}

std::unique_ptr<ExprAST> Parser::parseIdentifierExpr() {
  Location loc = lexer.getLastLoc();
  std::string name = lexer.getIdentifier();
  getNextToken();
  return std::make_unique<VariableExprAST>(loc, name);
}

std::unique_ptr<ExprAST> Parser::parseBinOpRHS(Location loc, int exprPrec,
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
      rhs = parseBinOpRHS(loc, tokPrec + 1, std::move(rhs));
      if (!rhs) return nullptr;
    }

    lhs = std::make_unique<BinaryExprAST>(loc, binOp, std::move(lhs),
                                          std::move(rhs));
  }
}

}  // namespace toy
