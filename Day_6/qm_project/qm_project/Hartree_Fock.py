import numpy as np
from copy import deepcopy

class Hartree_Fock:
    def __init__(self, atomic_coordinates, gas_model):
        self.atomic_coordinates = atomic_coordinates.copy()
        self.gas_model = deepcopy(gas_model)
        self.ndof = len(self.atomic_coordinates) * self.gas_model.orbitals_per_atom

        self.interaction_matrix = self.calculate_interaction_matrix()
        # chi tensor
        self.chi_tensor = self.calculate_chi_tensor()

        # Initial Hamiltonian and Density matrix
        self.hamiltonian_matrix = self.calculate_hamiltonian_matrix()
        self.density_matrix = self.calculate_atomic_density_matrix()
        self.fock_matrix = self.calculate_fock_matrix()
    
    def solve(self):
        # SCF Cycle
        self.scf_cycle()  
        energy_ion = self.calculate_energy_ion()
        energy_scf = self.calculate_energy_scf()
        
        return energy_scf + energy_ion
    
    def calculate_interaction_matrix(self):
        '''Calculates the electron-electron interaction energy matrix for an input list of atomic coordinates.
            
        Returns
        -------
        interaction_matrix : np.array
        '''

        interaction_matrix = np.zeros((self.ndof, self.ndof))
        for p in range(self.ndof):
            for q in range(self.ndof):
                if self.gas_model.atom(p) != self.gas_model.atom(q):
                    r_pq = self.atomic_coordinates[self.gas_model.atom(p)] - self.atomic_coordinates[self.gas_model.atom(
                        q)]
                    interaction_matrix[p, q] = self.coulomb_energy(self.gas_model.orb(p), self.gas_model.orb(q), r_pq)
                if p == q and self.gas_model.orb(p) == 's':
                    interaction_matrix[p, q] = self.gas_model.model_parameters['coulomb_s']
                if p == q and self.gas_model.orb(p) in self.gas_model.p_orbitals:
                    interaction_matrix[p, q] = self.gas_model.model_parameters['coulomb_p']
        return interaction_matrix

    def coulomb_energy(self, o1, o2, r12):
        '''Calculate the Coulomb energy for a pair of multipoles of type o1 & o2 separated by a vector r12.

        This function is used to build the Coulomb matrix.
        
        Parameters
        ----------
        o1 : str
            The type of orbital 1 (s, px, py, or pz)
        o2 : str
            The type of orbital 2 (s, px, py, or pz)
        r12 : np.array
            The distance vector between atoms for orbitals 1 and orbital 2.
        
        Return
        ------
        float
            The Coulomb energy
        '''
        r12_length = np.linalg.norm(r12)
        p_orbitals = self.gas_model.p_orbitals
        vec = self.gas_model.vec

        if o1 == 's' and o2 == 's':
            ans = 1.0 / r12_length
        if o1 == 's' and o2 in p_orbitals:
            ans = np.dot(vec[o2], r12) / r12_length**3
        if o2 == 's' and o1 in p_orbitals:
            ans = -1 * np.dot(vec[o1], r12) / r12_length**3
        if o1 in p_orbitals and o2 in p_orbitals:
            ans = (
                np.dot(vec[o1], vec[o2]) / r12_length**3 -
                3.0 * np.dot(vec[o1], r12) * np.dot(vec[o2], r12) / r12_length**5)
        return ans
    
    def calculate_potential_vector(self):
        '''Calculate the electron-ion potential energy vector for an input list of atomic coordinates.

        Returns
        -------
        np.array
            The potential vector
        '''
        potential_vector = np.zeros(self.ndof)
        for p in range(self.ndof):
            potential_vector[p] = 0.0
            for atom_i, r_i in enumerate(self.atomic_coordinates):
                r_pi = self.atomic_coordinates[self.gas_model.atom(p)] - r_i
                if atom_i != self.gas_model.atom(p):
                    potential_vector[p] += (
                        self.pseudopotential_energy(self.gas_model.orb(p), r_pi) -
                        self.gas_model.ionic_charge * self.coulomb_energy(self.gas_model.orb(p), 's', r_pi))
        return potential_vector
    
    def calculate_chi_tensor(self):
        '''Returns the chi tensor for an input list of atomic coordinates.
        
        Parameters
        ----------
        atomic_coordinates : np.array
            The atomic coordinates - format np.array([n,3]), where n is the number of atoms and the columns correspond to the x, y, and z coordinates.
        model_parameters : dict
            The model parameters for the element of interest.

        Returns
        -------
        np.array
            The chi tensor
        '''
        chi_tensor = np.zeros((self.ndof, self.ndof, self.ndof))
        for p in range(self.ndof):
            for orb_q in self.gas_model.orbital_types:
                q = self.gas_model.ao_index(self.gas_model.atom(p), orb_q)
                for orb_r in self.gas_model.orbital_types:
                    r = self.gas_model.ao_index(self.gas_model.atom(p), orb_r)
                    chi_tensor[p, q, r] = self.chi_on_atom(self.gas_model.orb(p), self.gas_model.orb(q), self.gas_model.orb(r))
        return chi_tensor

    def chi_on_atom(self, o1, o2, o3):
        '''Calculates the value of the chi tensor for 3 orbital indices on the same atom.

        This function calculates elements of the chi tensor matrix - used in `calculate_chi_tensor`
        
        Parameters
        ----------
        o1 : str
            The type of orbital 1 (s, px, py, or pz)
        o2 : str
            The type of orbital 2 (s, px, py, or pz)
        o3 : str 
            The type of orbital 3 (s, px, py, or pz)
        model_parameters: dict
            The model parameters for the element of interest
        
        Returns
        -------
        float
        '''
        p_orbitals = self.gas_model.p_orbitals
        dipole = self.gas_model.model_parameters['dipole']

        if o1 == o2 and o3 == 's':
            return 1.0
        if o1 == o3 and o3 in p_orbitals and o2 == 's':
            return dipole
        if o2 == o3 and o3 in p_orbitals and o1 == 's':
            return dipole
        return 0.0
    
    def hopping_energy(self, o1, o2, r12):
        '''Returns the hopping energy for a pair of orbitals of type o1 & o2 separated by a vector r12.

        This hopping energy is put into a hopping matrix, where o1 and o2 are used for indices. This is used in building the Hamiltonian matrix.
        
        Parameters
        ----------
        o1 : str
            The orbital type of the first orbital (s, px, py, or pz)
        o2 : str
            The orbital type of the second orbital (s, px, py, or pz)
        r12 : np.array
            The distance vector between atoms for orbitals 1 and orbital 2.
        model_parameters : dict
            A dictionary of model parameters for the calculation.

        Returns
        -------
        ans : float
            The hopping energy for orbitals o1 and o2.
        '''
        model_parameters = self.gas_model.model_parameters
        vec = self.gas_model.vec
        p_orbitals = self.gas_model.p_orbitals

        r12_rescaled = r12 / model_parameters['r_hop']
        r12_length = np.linalg.norm(r12_rescaled)
        ans = np.exp(1.0 - r12_length**2)
        if o1 == 's' and o2 == 's':
            ans *= model_parameters['t_ss']
        if o1 == 's' and o2 in p_orbitals:
            ans *= np.dot(vec[o2], r12_rescaled) * model_parameters['t_sp']
        if o2 == 's' and o1 in p_orbitals:
            ans *= -1 * np.dot(vec[o1], r12_rescaled) * model_parameters['t_sp']
        if o1 in p_orbitals and o2 in p_orbitals:
            ans *= ( (r12_length**2) * np.dot(vec[o1], vec[o2]) *
                    model_parameters['t_pp2'] -
                    np.dot(vec[o1], r12_rescaled) * np.dot(vec[o2], r12_rescaled) *
                    (model_parameters['t_pp1'] + model_parameters['t_pp2']))
        return ans

    def calculate_hamiltonian_matrix(self):
        '''Returns the 1-body Hamiltonian matrix for an input list of atomic coordinates.
        
        Returns
        -------
        np.array
            The Hamiltonian matrix
        
        '''
        hamiltonian_matrix = np.zeros((self.ndof, self.ndof))
        potential_vector = self.calculate_potential_vector()

        model_parameters = self.gas_model.model_parameters

        for p in range(self.ndof):
            for q in range(self.ndof):
                if self.gas_model.atom(p) != self.gas_model.atom(q):
                    r_pq = self.atomic_coordinates[self.gas_model.atom(p)] - self.atomic_coordinates[self.gas_model.atom(
                        q)]
                    hamiltonian_matrix[p, q] = self.hopping_energy(
                        self.gas_model.orb(p), self.gas_model.orb(q), r_pq)
                if self.gas_model.atom(p) == self.gas_model.atom(q):
                    if p == q and self.gas_model.orb(p) == 's':
                        hamiltonian_matrix[p, q] += model_parameters['energy_s']
                    if p == q and self.gas_model.orb(p) in self.gas_model.p_orbitals:
                        hamiltonian_matrix[p, q] += model_parameters['energy_p']
                    for orb_r in self.gas_model.orbital_types:
                        r = self.gas_model.ao_index(self.gas_model.atom(p), orb_r)
                        hamiltonian_matrix[p, q] += (
                            self.chi_on_atom(self.gas_model.orb(p), self.gas_model.orb(q), orb_r) *
                            potential_vector[r])
        return hamiltonian_matrix

    def pseudopotential_energy(self, o, r):
        '''Calculate the energy of a pseudopotential between a multipole of type o and an atom separated by a vector r.
        
        Parameters
        ----------
        o : str
            The atomic orbital type (s, px, py, or pz)
        r : np.array
            The vector from an atom to orbital o
        
        Returns
        -------
        float
            The pseudopotential energy
        '''
        model_parameters = self.gas_model.model_parameters

        ans = model_parameters['v_pseudo']
        r_rescaled = r / model_parameters['r_pseudo']
        ans *= np.exp(1.0 - np.dot(r_rescaled, r_rescaled))
        if o in self.gas_model.p_orbitals:
            ans *= -2.0 * np.dot(self.gas_model.vec[o], r_rescaled)
        return ans
    
    def calculate_atomic_density_matrix(self):
        '''Returns a trial 1-electron density matrix for an input list of atomic coordinates.
        
        Parameters
        ----------
        atomic_coordinates : np.array
            The atomic coordinates - format np.array([n,3]), where n is the number of atoms and the columns correspond to the x, y, and z coordinates.
        
        Returns
        -------
        np.array
        '''

        density_matrix = np.zeros((self.ndof, self.ndof))
        for p in range(self.ndof):
            density_matrix[p, p] = self.gas_model.orbital_occupation[self.gas_model.orb(p)]
        return density_matrix
    
    def calculate_fock_matrix(self):
        '''Returns the Fock matrix defined by the input Hamiltonian, interaction, & density matrices.'''

        return qm_cpp.calculate_fock_matrix_fast(self.hamiltonian_matrix, self.interaction_matrix, self.density_matrix, self.gas_model.model_parameters['dipole'])

    def calculate_density_matrix(self):
        '''Returns the 1-electron density matrix defined by the input Fock matrix.'''
        num_occ = (self.gas_model.ionic_charge // 2) * np.size(self.fock_matrix,
                                                0) // self.gas_model.orbitals_per_atom
        orbital_energy, orbital_matrix = np.linalg.eigh(self.fock_matrix)
        occupied_matrix = orbital_matrix[:, :num_occ]
        density_matrix = occupied_matrix @ occupied_matrix.T
        return density_matrix

    def scf_cycle(self):
        '''Returns converged density & Fock matrices defined by the input Hamiltonian, interaction, & density matrices.'''
        ## TODO - Give students this rewritten function.
        MAX_SCF_ITERATIONS = 100
        MIXING_FRACTION = 0.25
        CONVERGENCE_TOLERANCE = 1e-4
        for iteration in range(MAX_SCF_ITERATIONS):
            self.fock_matrix = self.calculate_fock_matrix()
            new_density_matrix = self.calculate_density_matrix()

            error_norm = np.linalg.norm(self.density_matrix - new_density_matrix)
            if error_norm < CONVERGENCE_TOLERANCE:
                return density_matrix

            density_matrix = (MIXING_FRACTION * new_density_matrix +
                            (1.0 - MIXING_FRACTION) * self.density_matrix)
            
            self.density_matrix = density_matrix

        print("SCF cycle didn't converge")
        return density_matrix
    
    def calculate_energy_ion(self):
        '''Calculate the ionic contribution to the total energy for an input list of atomic coordinates.
    
        Parameters
        ----------
        atomic_coordinates : np.array
            The atomic coordinates - format np.array([n,3]), where n is the number of atoms and the columns correspond to the x, y, and z coordinates.

        Returns
        -------
        float
            The ionic contribution to the total energy.
        '''
        energy_ion = 0.0
        for i, r_i in enumerate(self.atomic_coordinates):
            for j, r_j in enumerate(self.atomic_coordinates):
                if i < j:
                    energy_ion += (self.gas_model.ionic_charge**2) * self.coulomb_energy(
                        's', 's', r_i - r_j)
        return energy_ion

    def calculate_energy_scf(self):
        '''Returns the Hartree-Fock total energy defined by the input Hamiltonian, Fock, & density matrices.'''
        energy_scf = np.einsum('pq,pq', self.hamiltonian_matrix + self.fock_matrix,
                            self.density_matrix)
        return energy_scf
