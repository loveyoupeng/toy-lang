#ifndef TOY_AST_H_
#define TOY_AST_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace toy {

/// Supported numeric types in Toy.
enum class DataType {
  Byte,  // uint8
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt32,
  UInt64,
  Float32,
  Float64,
  Inferred
};

struct VarType {
  DataType type;
};

class ExprAST {
 public:
  virtual ~ExprAST() = default;
};

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;
  DataType type;

 public:
  explicit NumberExprAST(double Val, DataType type = DataType::Float64)
      : Val(Val), type(type) {}
  double getVal() const {
    return Val;
  }
  DataType getType() const {
    return type;
  }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

 public:
  explicit VariableExprAST(std::string Name) : Name(std::move(Name)) {}
  const std::string& getName() const {
    return Name;
  }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

 public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
  char getOp() const {
    return Op;
  }
  ExprAST* getLHS() const {
    return LHS.get();
  }
  ExprAST* getRHS() const {
    return RHS.get();
  }
};

/// Expression class for variable declarations.
class VarDeclAST : public ExprAST {
  std::string Name;
  VarType Type;
  std::unique_ptr<ExprAST> InitVal;
  bool isConstant;

 public:
  VarDeclAST(std::string Name, VarType Type, std::unique_ptr<ExprAST> InitVal,
             bool isConstant)
      : Name(std::move(Name)),
        Type(Type),
        InitVal(std::move(InitVal)),
        isConstant(isConstant) {}

  const std::string& getName() const {
    return Name;
  }
  const VarType& getType() const {
    return Type;
  }
  ExprAST* getInitVal() const {
    return InitVal.get();
  }
  bool IsConstant() const {
    return isConstant;
  }
};

/// A block of expressions (e.g. the whole file or a function body).
class BlockAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Expressions;

 public:
  explicit BlockAST(std::vector<std::unique_ptr<ExprAST>> Expressions)
      : Expressions(std::move(Expressions)) {}
  const std::vector<std::unique_ptr<ExprAST>>& getExpressions() const {
    return Expressions;
  }
};

}  // namespace toy

#endif  // TOY_AST_H_
