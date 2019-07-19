# Notes on the QM project

`orbital_types` - this is specific to Argon. 

Our simulation will only work for Argon.
- atomic_coordinates can be specified as user input
- orbital_types and derived quantities are specific for Argon and can be specified in a metadata file.

~~~
orbital_types = ['s', 'px', 'py', 'pz']
orbitals_per_atom = len(orbital_types)
p_orbitals = orbital_types[1:]
~~~

Things to put in tests which are in the jupyter notebook

~~~
print(atomic_coordinates)

print("1st atom index =", atom(0))
print("1st orbital type =", orb(0))
print("ao index of s orbital on 1st atom =", ao_index(0, 's'))
~~~

~~~

# This is a test
print("vec[px] =", vec['px'])
print("hopping test",
    hopping_matrix_element('s', 'px', np.array([1.0, 0.0, 0.0])))
~~~

