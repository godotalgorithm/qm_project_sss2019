import pytest
import numpy as np

import day_2 as qm

@pytest.fixture
def argon_model_parameters():
    model_parameters = {
    'r_hop': 5.0,
    't_ss': -0.002,
    't_sp': -0.004,
    't_pp1': -0.006,
    't_pp2': -0.008,
    'r_pseudo': 6.0,
    'v_pseudo': 0.5,
    'dipole': 4.0,
    'energy_s': 1.3,
    'energy_p': 0.1,
    'self_energy': 0.5
    }
    return model_parameters

@pytest.mark.parametrize("ao_index, atom_index", [
    (0, 0),
    (1, 0),
    (4, 1),
    (5, 1)
])
def test_atom_index(ao_index, atom_index):
    assert qm.atom(ao_index) == atom_index

@pytest.mark.parametrize("ao_index, orbital", [
    (0, 's'),
    (1, 'px'),
    (2, 'py'),
    (3, 'pz'),
    (4, 's')
])
def test_orb(ao_index, orbital):
    assert qm.orb(ao_index) == orbital

@pytest.mark.parametrize("atom_p, orb_p, orbital_index",[
    (0, 's', 0),
    (0, 'px', 1),
    (0, 'py', 2),
    (0, 'pz', 3),
    (1, 'px', 5)
])
def test_ao_index(atom_p, orb_p, orbital_index):
    assert qm.ao_index(atom_p, orb_p) == orbital_index

def test_hopping_energy(argon_model_parameters):
    orbital1 = 's'
    orbital2 = 'px'
    orbital_distance = np.array([5.0, 0.0, 0.0])
    assert qm.hopping_energy(orbital1, orbital2, orbital_distance, argon_model_parameters) == -0.004

@pytest.mark.parametrize("o1, o2, r12, coulomb_energy",[
    ('s', 's', np.array([1, 0, 0]), 1),
    ('s', 'px', np.array([1, 0, 0]), 1),
    ('px', 's', np.array([1, 0, 0]), -1),
    ('py', 'px', np.array([1, 0, 0]), 0),
])
def test_coulomb_energy(o1, o2, r12, coulomb_energy):
    assert qm.coulomb_energy(o1, o2, r12) == coulomb_energy

def test_energy_ion():
    atomic_coordinates = np.array([ [0.0,0.0,0.0], [1.0,0.0,0.0] ])
    energy_ion = qm.calculate_energy_ion(atomic_coordinates)
    assert energy_ion == 36

def test_scf_cycle(argon_model_parameters):
    atomic_coordinates = np.array([ [0.0,0.0,0.0], [3.0,4.0,5.0] ])
    hamiltonian_matrix = qm.calculate_hamiltonian_matrix(atomic_coordinates, argon_model_parameters)
    interaction_matrix = qm.calculate_interaction_matrix(atomic_coordinates, argon_model_parameters)
    density_matrix = qm.calculate_atomic_density_matrix(atomic_coordinates)
    chi_tensor = qm. calculate_chi_tensor(atomic_coordinates, argon_model_parameters)
    fock_matrix = qm.calculate_fock_matrix(hamiltonian_matrix, interaction_matrix, density_matrix, chi_tensor)
    
    density_matrix, fock_matrix = qm.scf_cycle(hamiltonian_matrix, interaction_matrix, density_matrix, chi_tensor)

    energy_ion = qm.calculate_energy_ion(atomic_coordinates)

    calculated_energy = qm.calculate_energy_scf(hamiltonian_matrix, fock_matrix, density_matrix) + energy_ion
    
    assert np.isclose(calculated_energy, -40.23578207295311)

