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
        for a in range(num_virt):
            for b in range(num_virt):
                for i in range(num_occ):
                    for j in range(num_occ):
                        energy_mp2 -= (
                            (2.0 * V_tilde[a, i, b, j]**2 -
                            V_tilde[a, i, b, j] * V_tilde[a, j, b, i]) /
                            (E_virt[a] + E_virt[b] - E_occ[i] - E_occ[j]))
        return energy_mp2
