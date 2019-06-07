# 2019 Software Summer School - QM Sample Repo
A sample repo for the software summer school project.

Each day will have milestones for the students to complete. 

## Day 1

### Summary
On the first day, students will implement a Hartree-Fock & many-body perturbation theory script individually (following the QM lead, Jonathan). After a working script is implemented, students will work individually to complete the student milestones outlined here.

### Instructor Milestones
It has been decided that the Instructor Script will have the following qualities:

1. The instructor script will define a semiempirical model for a cluster of Argon atoms.
1. The instructor script will implement Hartree-Fock and many-body perturbation theory calculations on the model.
1. The instructor script will be a flat script with no user defined functions.
1. The instructor script will use Hartree atomic units.
1. The instructor script will read an initial atomic configuration from an xyz file.
1. The instructor script will store all matrix elements in memory at the start of the caculation.
1. The instructor script will perform an SCF cycle with simple density-matrix mixing.
1. The instructor script will implement the textbook 2nd-order Moller-Plesset perturbation theory (MP2) formula.
1. The instructor script will contain reference data for students to check the model against.
1. The instructor script will contain basic timing functions to assess the scaling of algorithms.
1. The instructor script will contain a localized MP2 formula so that students can see the benefits of special-purpose fast algorithms compared with other optimizations/refactorizations that they might perform.

### Student Milestones
The student will refactor the Instructor Script into functions.

1. The semiempirical model should be isolated inside of functions
1. Blocks of matrix elements should be functions to avoid saving them in memory (i.e. "direct" QM methods).
1. The SCF cycle should be a function so that a good initial density matrix can be specified when refining solutions.
1. The standard and localized MP2 calculations should be functions to facilitate cost & accuracy benchmarks between them.
1. The entire calculation should be performable for a given set of coordinates so that the model can be checked against the provided reference data.
1. **Extension**: further modification of the localized MP2 calculation to truncate distant pairs using a precomputed neighbor list.
