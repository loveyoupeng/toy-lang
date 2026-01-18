if(benchmark_ADDED)
  add_executable(toy_benchmark bench_main.cpp)
  target_link_libraries(toy_benchmark PRIVATE benchmark::benchmark_main)
endif()
