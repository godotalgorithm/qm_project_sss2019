from qm_project.Hartree_Fock import Hartree_Fock
import numpy as np

class MP2(Hartree_Fock):
    def __init__(self, atomic_coordinates, gas_model):
        super().__init__(atomic_coordinates, gas_model)

    def solve(self):
        self.hf_energy = super().solve()
        self.occupied_energy, self.virtual_energy, self.occupied_matrix, self.virtual_matrix = self.partition_orbitals()
        self.interaction_tensor = self.transform_interaction_tensor()
        return self.calculate_energy_mp2()
    
    
    def partition_orbitals(self):
        '''Returns a list with the occupied/virtual energies & orbitals defined by the input Fock matrix.'''
        num_occ = (self.gas_model.ionic_charge // 2) * np.size(self.fock_matrix,
                                                0) // self.gas_model.orbitals_per_atom
        orbital_energy, orbital_matrix = np.linalg.eigh(self.fock_matrix)
        occupied_energy = orbital_energy[:num_occ]
        virtual_energy = orbital_energy[num_occ:]
        occupied_matrix = orbital_matrix[:, :num_occ]
        virtual_matrix = orbital_matrix[:, num_occ:]

        return occupied_energy, virtual_energy, occupied_matrix, virtual_matrix
    
    def transform_interaction_tensor(self):
        '''Returns a transformed V tensor defined by the input occupied, virtual, & interaction matrices.'''
        chi2_tensor = np.einsum('qa,ri,qrp',
                                self.virtual_matrix,
                                self.occupied_matrix,
                                self.chi_tensor,
                                optimize=True)
        interaction_tensor = np.einsum('aip,pq,bjq->aibj',
                                    chi2_tensor,
                                    self.interaction_matrix,
                                    chi2_tensor,
                                    optimize=True)
        return interaction_tensor
    
    def calculate_energy_mp2(self):
        '''Returns the MP2 contribution to the total energy defined by the input Fock & interaction matrices.'''
        E_occ, E_virt, self.occupied_matrix, self.virtual_matrix = self.partition_orbitals()
        V_tilde = self.transform_interaction_tensor()

        energy_mp2 = 0.0
        num_occ = len(E_occ)
        num_virt = len(E_virt)
        V_tilde_flat = np.reshape(V_tilde, (num_occ*num_virt, num_occ*num_virt))

        return qm_cpp.calculate_energy_mp2(E_occ, E_virt, V_tilde_flat)
