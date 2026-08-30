add_executable(test_algorithms test_algorithms.cpp)

# Link against GTest and our graph libraries
target_link_libraries(test_algorithms 
    PRIVATE 
    gtest_main 
    graph_core 
    graph_algorithms
    OpenMP::OpenMP_CXX
)

# Allow GTest to find our headers
target_include_directories(test_algorithms PRIVATE ${CMAKE_SOURCE_DIR}/include)

include(GoogleTest)
gtest_discover_tests(test_algorithms)
