#ifndef TOY_AST_H
#define TOY_AST_H

#include <memory>
#include <vector>
#include <string>

namespace toy {

class ExprAST {
public:
  virtual ~ExprAST() = default;
};

class NumberExprAST : public ExprAST {
  double Val;
public:
  NumberExprAST(double Val) : Val(Val) {}
  double getVal() const { return Val; }
};

class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;
public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS, std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  char getOp() const { return Op; }
  ExprAST* getLHS() const { return LHS.get(); }
  ExprAST* getRHS() const { return RHS.get(); }
};

} // namespace toy

#endif // TOY_AST_H
