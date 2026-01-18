#include <gtest/gtest.h>

TEST(ToyLangTest, BasicAssertions) {
  EXPECT_STRNE("hello", "world");
  EXPECT_EQ(7 * 6, 42);
}
