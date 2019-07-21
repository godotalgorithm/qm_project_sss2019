import numpy as np
from copy import deepcopy

class Noble_Gas_Model:
    def __init__(self, gas_type):
        if ( gas_type == 'Argon' ):
            self.model_parameters = {
                'r_hop' : 3.1810226927827516,
                't_ss' : 0.03365982238611262,
                't_sp' : -0.029154833035109226,
                't_pp1' : -0.0804163845390335,
                't_pp2' : -0.01393611496959445,
                'r_pseudo' : 2.60342991362958,
                'v_pseudo' : 0.022972992186364977,
                'dipole' : 2.781629275106456,
                'energy_s' : 3.1659446174413004,
                'energy_p' : -2.3926873325346554,
                'coulomb_s' : 0.3603533286088998,
                'coulomb_p' : -0.003267991835806299
                }
        elif ( gas_type == 'Neon' ):
            self.model_parameters = {
                'coulomb_p': -0.010255409806855187,
                'coulomb_s': 0.4536486561938202,
                'dipole': 1.6692376991516769,
                'energy_p': -3.1186533988406335,
                'energy_s': 11.334912902362603,
                'r_hop': 2.739689713337267,
                'r_pseudo': 1.1800779720963734,
                't_pp1': -0.029546671673199854,
                't_pp2': -0.0041958662271044875,
                't_sp': 0.000450562836426027,
                't_ss': 0.0289251941290921,
                'v_pseudo': -0.015945813280635074
                }
        else:
            raise TypeError('Gas type ' + gas_type + ' not recognized.')
            
        self.ionic_charge = 6
        self.orbital_types = ['s', 'px', 'py', 'pz']
        self.p_orbitals = [x for x in self.orbital_types if 'p' in x ]
        self.orbitals_per_atom = len(self.orbital_types)
        self.p_orbitals = self.orbital_types[1:]
        self.vec = {'px': [1, 0, 0], 'py': [0, 1, 0], 'pz': [0, 0, 1]}
        self.orbital_occupation = { 's':0, 'px':1, 'py':1, 'pz':1 }

    def atom(self, ao_index):
        '''Calculates the atom index part of an atomic orbital index.
    
        Given an atomic orbital index, this function will return the index of the atom. Can be used to index into the atomic_coordinates array.
        
        Parameters
        ----------
        ao_index : int
            The atomic orbital index

        Returns
        -------
        int
            The atom index
         '''

        return ao_index // self.orbitals_per_atom
    
    def ao_index(self, atom_p, orb_p):
        '''Returns the atomic orbital index for a given atom index and orbital type.
    
        Parameters
        ----------
        atom_p : int
            The atom index
        orb_p : str
            The orbital type. Uses `orbital_types` - can be `px`, `py`, `pz`.

        Returns
        -------
        int
            The atomic orbital index
        '''
        
        p = atom_p * self.orbitals_per_atom
        p += self.orbital_types.index(orb_p)
        return p
    
    def orb(self, ao_index):
        '''Returns the orbital type of an atomic orbital index.
        
        Parameters
        ----------
        ao_index : int
            The atomic orbital index

        Returns
        -------
        str
            The atomic orbital type. 
        '''
        orb_index = ao_index % self.orbitals_per_atom
        return self.orbital_types[orb_index]
    
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
                r_pi = atomic_coordinates[self.gas_model.atom(p)] - r_i
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
        fock_matrix = self.hamiltonian_matrix.copy()
        fock_matrix += 2.0 * np.einsum('pqt,rsu,tu,rs',
                                    self.chi_tensor,
                                    self.chi_tensor,
                                    self.interaction_matrix,
                                    self.density_matrix,
                                    optimize=True)
        fock_matrix -= np.einsum('rqt,psu,tu,rs',
                                self.chi_tensor,
                                self.chi_tensor,
                                self.interaction_matrix,
                                self.density_matrix,
                                optimize=True)
        return fock_matrix

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

    

if __name__ == "__main__":

    # User input
    atomic_coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])

    # Create Noble Gas Model
    argon_model = Noble_Gas_Model('Argon')

    hf = MP2(atomic_coordinates, argon_model)
    # Hartree Fock Energy
    print(hf.solve())

   

    


    

