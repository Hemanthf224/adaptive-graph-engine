# Create a core graph library
add_library(graph_core STATIC graph.cpp io.cpp scheduler.cpp query_parser.cpp bytecode_vm.cpp)
target_link_libraries(graph_core PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX CUDA::cudart)
target_include_directories(graph_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Create algorithms library
add_library(graph_algorithms STATIC bfs.cpp bfs_cuda.cu page_rank.cpp page_rank_cuda.cu page_rank_mpi.cpp connected_components.cpp connected_components_cuda.cu sssp.cpp partitioning.cpp)
target_link_libraries(graph_algorithms PUBLIC graph_core)

# Main executable
add_executable(graph_engine main.cpp)
target_link_libraries(graph_engine PRIVATE graph_core graph_algorithms OpenMP::OpenMP_CXX MPI::MPI_CXX)


