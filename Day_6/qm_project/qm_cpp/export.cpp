#include "day_6.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"


PYBIND11_MODULE(qm_cpp, m)
{
    m.doc() = "Functions for the QM project implemented in C++";
    
    m.def("calculate_fock_matrix_fast", calculate_fock_matrix_fast,
          "Calculate the fock matrix");

    m.def("calculate_energy_mp2", calculate_energy_mp2,
          "Calculate the MP2 energy");
}
