
from qm_project.Hartree_Fock import Hartree_Fock
from qm_project.MP2 import MP2
from qm_project.Noble_Gas_Model import Noble_Gas_Model

def run(coordinates, element='Argon', method='MP2'):
    model = Noble_Gas_Model(element)

    if method.lower() == 'mp2':
        calc = MP2(coordinates, model)
    elif method.lower() == 'hf':
        calc = Hartree_Fock(coordinates, model)
    else:
        raise RuntimeError('Invalid method: ' + method)

    print(calc.solve())
