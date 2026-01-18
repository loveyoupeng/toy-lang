if(googletest_ADDED)
  add_executable(toy_test 
    test_main.cpp
    TypeCheckerTest.cpp
  )
  target_link_libraries(toy_test PRIVATE 
    toy_compiler
    GTest::gtest_main
  )
  
  include(GoogleTest)
  gtest_discover_tests(toy_test)
endif()