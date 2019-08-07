#pragma once

#include <vector>
#include <Eigen/Dense>


/* We use the following indices for orbital types:
   0 = s
   1 = px
   2 = py
   3 = pz

   Therefore, if an index is > 0, it is a p orbital
*/

// This is the number of orbital types we have
const int orbitals_per_atom = 4;



/*! \brief Calculate a fock matrix */
Eigen::MatrixXd calculate_fock_matrix_fast(Eigen::MatrixXd hamiltonian_matrix,
                                           Eigen::MatrixXd interaction_matrix,
                                           Eigen::MatrixXd density_matrix,
                                           double model_dipole);

/*! \brief Calculate MP2 energy */
double calculate_energy_mp2(std::vector<double> E_occ, std::vector<double> E_virt, Eigen::MatrixXd v_tilde_flat);
