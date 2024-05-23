'''
Solves the TISE for the double well potential.
'''
import sys
import numpy as np
from time import process_time
from copy import copy
from tise.grid import create_grid
from tise.hamilton import create_hamiltonian
from tise.eigen import get_smallest_eigenvalue
from tise.observables import mu_O, var_O, apply_p_1d, apply_x_1d, sphere_r_prob


def read_parameters_from_file(path: str, params: dict):
    '''
    Reads arbitrary parameters into a dictionary. The input file
    has to be of the format:

    key1=value1
    key2=value2

    path: The path of the file to read
    params: A key:constructor parameter template dictionary.
            Only variables specified in params  will be read,
            they will be read in by the constructor specified in params
    returns: A dictionary of variables read from path
    ...
    '''
    with open(path, "r") as fd:
        content = fd.read()
        print(content)
        content = content.replace('\n', '')
        lines = content.split(';')
        for line in lines:
            if len(line) == 0:
                continue
            key, value = line.split('=')
            # print(key, value)
            if key in params.keys():
                params[key] = params[key](value)
    return params


def write_parameters_to_file(path: str, params: dict):
    '''
    Writes arbitrary parameters in a dictionary to a file.
    The resulting file will be of the format

    key1=value1
    key2=value2
    ...

    And is therefor parsable by 'read_parameters_from_file'
    path: The path of the file to read
    params: A key:constructor parameter template dictionary.
            Only variables specified in params  will be read,
            they will be read in by the constructor specified in params
    '''
    with open(path, 'w') as fd:
        for key, value in params.items():
            fd.write(f'{key}={value};\n')


if __name__ == "__main__":
    filenames = sorted(sys.argv[1:])
    parameter_template = {
        'eps': float,
        'mu': float,
        'N': int,
        'D': int,
        'tolerance_cg': float,
        'tolerance_pm': float,
        'res_recal_iterations': int,
    }
    if len(filenames) == 0:
        print("At least one filename has to be specified")
    for filename in filenames:
        parameters = copy(parameter_template)
        parameters = read_parameters_from_file(filename, parameters)
        grid = create_grid(parameters['N'], parameters['D'])
        hamiltonian = create_hamiltonian(grid, parameters['mu'], parameters['eps'])
        test_vector = np.random.random(size=(2*parameters['N']+1, )*parameters['D'])
        inv_slice = (slice(None, None, -1), )*parameters['D']
        test_vector = test_vector + test_vector[inv_slice]
        print('calculating for parameters in file: ' + filename)
        print(parameters)
        start = process_time()
        energy, state, _ = get_smallest_eigenvalue(hamiltonian, test_vector,
                                                   parameters['tolerance_cg'], parameters['tolerance_pm'],
                                                   parameters['res_recal_iterations'])
        stop = process_time()
        parameters['energy'] = energy
        R = 1 / parameters['eps']
        mu_x = []
        mu_p = []
        sig_x = []
        sig_p = []
        for axis in range(parameters['D']):
            mu_x.append(mu_O(state, lambda x: apply_x_1d(x, axis, R)))
            sig_x.append(np.sqrt(var_O(state, lambda x: apply_x_1d(x, axis, R))))
            mu_p.append(mu_O(state, lambda p: apply_p_1d(p, axis, R)))
            sig_p.append(np.sqrt(var_O(state, lambda p: apply_p_1d(p, axis, R))))
        parameters['mu_x'] = mu_x
        parameters['mu_p'] = mu_p
        parameters['sig_x'] = sig_x
        parameters['sig_p'] = sig_p
        parameters['Propability_in_R'] = sphere_r_prob(state, R)
        parameters['Process time for simulation'] = stop - start
        write_parameters_to_file(filename, parameters)
        np.save(filename + '_state', state)
