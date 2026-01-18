#ifndef TOY_AST_H_
#define TOY_AST_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace toy {

struct Location {
  int line;
  int col;
};

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
  Location Loc;

 public:
  explicit ExprAST(Location Loc) : Loc(Loc) {}
  virtual ~ExprAST() = default;
  const Location& loc() const {
    return Loc;
  }
};

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;
  DataType type;

 public:
  NumberExprAST(Location Loc, double Val, DataType type = DataType::Float64)
      : ExprAST(Loc), Val(Val), type(type) {}
  double getVal() const {
    return Val;
  }
  DataType getType() const {
    return type;
  }
};

/// Expression class for string literals.
class StringExprAST : public ExprAST {
  std::string Val;

 public:
  StringExprAST(Location Loc, std::string Val)
      : ExprAST(Loc), Val(std::move(Val)) {}
  const std::string& getValue() const {
    return Val;
  }
};

/// Expression class for print/println.
class PrintExprAST : public ExprAST {
  bool isNewLine;
  bool toStderr;
  std::vector<std::unique_ptr<ExprAST>> args;

 public:
  PrintExprAST(Location Loc, bool isNewLine, bool toStderr,
               std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Loc),
        isNewLine(isNewLine),
        toStderr(toStderr),
        args(std::move(args)) {}

  bool getIsNewLine() const {
    return isNewLine;
  }
  bool getToStderr() const {
    return toStderr;
  }
  const std::vector<std::unique_ptr<ExprAST>>& getArgs() const {
    return args;
  }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

 public:
  VariableExprAST(Location Loc, std::string Name)
      : ExprAST(Loc), Name(std::move(Name)) {}
  const std::string& getName() const {
    return Name;
  }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;

 public:
  BinaryExprAST(Location Loc, char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : ExprAST(Loc), Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
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
  VarDeclAST(Location Loc, std::string Name, VarType Type,
             std::unique_ptr<ExprAST> InitVal, bool isConstant)
      : ExprAST(Loc),
        Name(std::move(Name)),
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

/// Expression class for explicit casts/macros like asint16(b).
class CastExprAST : public ExprAST {
  DataType DestType;
  std::unique_ptr<ExprAST> Arg;

 public:
  CastExprAST(Location Loc, DataType DestType, std::unique_ptr<ExprAST> Arg)
      : ExprAST(Loc), DestType(DestType), Arg(std::move(Arg)) {}

  DataType getDestType() const {
    return DestType;
  }
  ExprAST* getArg() const {
    return Arg.get();
  }
};

/// A block of expressions (e.g. the whole file or a function body).
class BlockAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> Expressions;

 public:
  BlockAST(Location Loc, std::vector<std::unique_ptr<ExprAST>> Expressions)
      : ExprAST(Loc), Expressions(std::move(Expressions)) {}
  const std::vector<std::unique_ptr<ExprAST>>& getExpressions() const {
    return Expressions;
  }
};

}  // namespace toy

#endif  // TOY_AST_H_
