#include <gtest/gtest.h>
#include "toy/AST.h"
#include "toy/Lexer.h"
#include "toy/Parser.h"
#include "toy/MLIRGen.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace toy;

class TypeCheckerTest : public ::testing::Test {
protected:
    mlir::MLIRContext context;
    void SetUp() override {
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();
    }

    std::unique_ptr<BlockAST> parse(const std::string& code) {
        Lexer lexer(code);
        Parser parser(std::move(lexer));
        return parser.parse();
    }
};

TEST_F(TypeCheckerTest, PromotionInt32ToInt64) {
    auto ast = parse("var a:int32 = 10; var b:int64 = a;");
    ASSERT_NE(ast, nullptr);
    auto module = mlirGen(context, *ast);
    EXPECT_NE(module, nullptr);
}

TEST_F(TypeCheckerTest, IllegalTruncationInt64ToInt32) {
    auto ast = parse("var a:int64 = 10; var b:int32 = a;");
    ASSERT_NE(ast, nullptr);
    // This should fail in codegen due to illegal implicit truncation
    auto module = mlirGen(context, *ast);
    EXPECT_EQ(module, nullptr);
}

TEST_F(TypeCheckerTest, ByteIsolation) {
    auto ast = parse("var a:byte = bx10; var b:int32 = a;");
    ASSERT_NE(ast, nullptr);
    auto module = mlirGen(context, *ast);
    EXPECT_EQ(module, nullptr);
}

TEST_F(TypeCheckerTest, ExplicitByteCast) {
    auto ast = parse("var a:byte = bx10; var b:int32 = asint32(a);");
    ASSERT_NE(ast, nullptr);
    auto module = mlirGen(context, *ast);
    EXPECT_NE(module, nullptr);
}

TEST_F(TypeCheckerTest, ComplexExpressionPromotion) {
    auto ast = parse("var a:int32 = 10; var b:float64 = 2.0; var c = a + b;");
    ASSERT_NE(ast, nullptr);
    auto module = mlirGen(context, *ast);
    EXPECT_NE(module, nullptr);
}
