#include <iostream>
#include "Solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

#include <chrono>

#include <tuple>

namespace py = pybind11;

using namespace Eigen;
using namespace std;


PYBIND11_MODULE(sappy_solver, m) {
    m.doc() = "Python bindings for C++/Eigen QCQP solver, implmented in primal coordinates via the SAP solver developed by Alejandro Castro et al. at TRI.";
    py::class_<Solver>(m, "Solver")
        .def(py::init<int,int>())
        .def("solve", &Solver::solve, "Solve SAP for given problem parameters.",py::arg("J"), py::arg("q"), py::arg("eps"));

}
