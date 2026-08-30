#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "core/graph.hpp"
#include "core/io.hpp"
#include "algorithms/page_rank_cuda.cuh"
#include "algorithms/bfs_cuda.cuh"
#include <omp.h>
#include <mpi.h>

namespace py = pybind11;
using namespace graph_engine;

// Need a simple struct wrapper or just use the CSRGraph directly
// Since uvm_vector inherits/is std::vector, pybind11::stl can handle it
// BUT copying back and forth between Python Lists and uvm_vector might be slow.
// For now, we will let Pybind11 do the standard stl conversions.

PYBIND11_MODULE(adaptive_graph, m) {
    m.doc() = "Adaptive GPU-Accelerated Graph Processing Engine Python API";

    // Expose the CSRGraph struct
    py::class_<core::CSRGraph>(m, "CSRGraph")
        .def(py::init<>())
        .def_readwrite("num_vertices", &core::CSRGraph::num_vertices)
        .def_readwrite("num_edges", &core::CSRGraph::num_edges)
        .def("print", &core::CSRGraph::print, "Print graph stats");

    // Expose IO
    m.def("load", &core::load_graph, "Load a graph from a Matrix Market or Edgelist file",
          py::arg("filepath"));

    // Expose CUDA Algorithms
    m.def("pagerank_cuda", &algorithms::pagerank_cuda, 
          "Run PageRank on the GPU using Zero-Copy UVM",
          py::arg("graph"), py::arg("iterations") = 20, py::arg("damping") = 0.85f);
          
    m.def("bfs_cuda", &algorithms::bfs_cuda, 
          "Run BFS on the GPU using Zero-Copy UVM",
          py::arg("graph"), py::arg("source") = 0);
}
