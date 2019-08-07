#include <Eigen/Dense>
#include <vector>
#include <cassert>
#include <cmath>
#include "day_6.hpp"


// We will be using these a lot so for convenience
// make it so we don't have to put Eigen:: in front of them.
using Eigen::MatrixXd;
using Eigen::Vector3d;



/*! \brief Returns the value of the chi tensor for 3 orbital indices on the same atom
 */
double chi_on_atom(int o1, int o2, int o3, double model_dipole)
{
    if(o1 == o2 and o3 == 0)
        return 1.0;
    if(o1 == o3 && o3 > 0 && o2 == 0)
        return model_dipole;
    if(o2 == o3 && o3 > 0 && o1 == 0)
        return model_dipole;
    return 0.0;
}


/*! \brief Returns the atom index part of an atomic orbital index
 */
int atom(int ao_index)
{
    // Division of two integers always results in an integer in C++
    return ao_index / orbitals_per_atom;
}


/*! \brief Returns the atomic orbital index for a given atom index and orbital type
 */
int ao_index(int atom_p, int orb_p)
{
    return atom_p*orbitals_per_atom + orb_p;
}


/*! \brief Returns the orbital type of an atomic orbital index
 */
int orb(int ao_index)
{
    return ao_index % orbitals_per_atom;
}


MatrixXd calculate_fock_matrix_fast(MatrixXd hamiltonian_matrix,
                                    MatrixXd interaction_matrix,
                                    MatrixXd density_matrix,
                                    double model_dipole)
{
    // Number of degrees of freedon
    const size_t ndof = hamiltonian_matrix.rows();
    MatrixXd fock_matrix(hamiltonian_matrix); // Calls copy constructor


    // Potential term
    for(size_t p = 0; p < ndof; p++)
    {
        for(int orb_q = 0; orb_q < orbitals_per_atom; orb_q++)
        {
            int q = ao_index(atom(p), orb_q); // p & q on same atom

            for(int orb_t = 0; orb_t < orbitals_per_atom; orb_t++)
            {
                int t = ao_index(atom(p), orb_t); // p & t on same atom
                double chi_pqt = chi_on_atom(orb(p), orb_q, orb_t, model_dipole);

                for(size_t r = 0; r < ndof; r++)
                {
                    for(int orb_s = 0; orb_s < orbitals_per_atom; orb_s++)
                    {
                        int s = ao_index(atom(r), orb_s); // r & s on same atom
                        for(int orb_u = 0; orb_u < orbitals_per_atom; orb_u++)
                        {
                            int u = ao_index(atom(r), orb_u); // r & u on same atom
                            double chi_rsu = chi_on_atom(orb(r), orb_s, orb_u, model_dipole);
                            fock_matrix(p,q) += 2.0 * chi_pqt * chi_rsu * interaction_matrix(t,u) * density_matrix(r,s);
                        }
                    }
                }
            }
        }
    }

    // Exchange term
    for(size_t p = 0; p < ndof; p++)
    {
        for(int orb_s = 0; orb_s < orbitals_per_atom; orb_s++)
        {
            int s = ao_index(atom(p), orb_s); // p & s on same atom
            for(int orb_u = 0; orb_u < orbitals_per_atom; orb_u++)
            {
                int u = ao_index(atom(p), orb_u); // p & u on same atom
                double chi_psu = chi_on_atom(orb(p), orb_s, orb_u, model_dipole);

                for(size_t q = 0; q < ndof; q++)
                {
                    for(int orb_r = 0; orb_r < orbitals_per_atom; orb_r++)
                    {
                        int r = ao_index(atom(q), orb_r); // q & r on same atom
                        for(int orb_t = 0; orb_t < orbitals_per_atom; orb_t++)
                        {
                            int t = ao_index(atom(q), orb_t);
                            double chi_rqt = chi_on_atom(orb_r, orb(q), orb_t, model_dipole);
                            fock_matrix(p,q) -= chi_rqt * chi_psu * interaction_matrix(t,u) * density_matrix(r,s);
                        }
                    }
                }
            }
        }
    }

    return fock_matrix;
}


double calculate_energy_mp2(std::vector<double> E_occ, std::vector<double> E_virt, MatrixXd v_tilde_flat)
{
    size_t num_occ = E_occ.size();    
    size_t num_virt = E_virt.size();    

    double energy_mp2 = 0.0;

    // The v_tilde matrix has been flattened from 4 indices to 2 indices.
    // The dimensions of the flattened tensor should be nvirt*nocc x nvirt*nocc
    size_t nrow = v_tilde_flat.rows(); // should be nvirt*nocc
    size_t ncol = v_tilde_flat.cols(); // should be nvirt*nocc
    assert(nrow == num_virt*num_occ);
    assert(ncol == num_virt*num_occ);

    for(size_t a = 0; a < num_virt; a++)
    for(size_t i = 0; i < num_occ; i++)
    {
        size_t idx_1 = a*num_occ+i;
        for(size_t b = 0; b < num_virt; b++)
        for(size_t j = 0; j < num_occ; j++)
        {
            size_t idx_2 = b*num_occ+j;
            size_t idx_3 = a*num_occ+j;
            size_t idx_4 = b*num_occ+i;

            double numerator = 2.0 * pow(v_tilde_flat(idx_1, idx_2), 2) - v_tilde_flat(idx_1, idx_2)*v_tilde_flat(idx_3, idx_4);
            double denominator = E_virt[a] + E_virt[b] - E_occ[i] - E_occ[j];
            energy_mp2 -= numerator/denominator;
        }
    }

    return energy_mp2;
}

