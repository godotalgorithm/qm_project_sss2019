# Day 2 

Students will work with their teams on a common repository (QM_teamXX_2019) to fulfill the following milestones. Changes to the repo should be done using a Fork/PR model, where every change must be reviewed by one other person before merging.

## Testing 
Write unit tests for your code. 

1. There are several places where values were printed to test the functions. You can use these as a template to start writing tests.

1. Create a pytest fixture to return model parameters.

1. Use `@pytest.mark.parametrize` to test `atom`, `orb` and `ao_index`.

1. You can use pytest parametrize on any function which uses a piecewise equation to test every case. For example, to test the `coulomb_energy` function, you could set up something like the following:

```
@pytest.mark.parametrize("o1, o2, r12, coulomb_energy",[
    ('s', 's', np.array([1, 0, 0]), 1),
    ('s', 'px', np.array([1, 0, 0]), 1), 
    ...
```

where each line corresponds to a different condition. 

1. Write a test which checks the calculation of the Hartree Fock energy.

1. Write a test which checks the calculation of the MP2 energy.

## Docstrings
1. Write [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings for you functions. 
