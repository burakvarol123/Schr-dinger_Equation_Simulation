'''
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from tise.observables import calculate_mean_position
from tdse.main import read_data


if __name__ == "__main__":
    B = int(sys.argv[1])
    # with open(sys.argv[1], 'r') as fd:
    #     tokens = fd.read().split(',')
    #     tokens = list(filter(lambda x: len(x) > 0, tokens))
    #     dts = tokens
    #     # dts = list(map(float, tokens))
    #     # dts.sort(reverse=True)
    dts = ['3e-3','1e-3','3e-4','1e-4','3e-5','1e-5']
    dts_float = list(map(float, dts))
    print(dts)

    euler_fns = [f'eu_{dt}.npy' for dt in dts]
    cranic_fns = [f'cn_{dt}.npy' for dt in dts]
    strang_fns = [f'ss_{dt}.npy' for dt in dts]

    euler_data = []
    cranic_data = []
    strang_data = []
    for i, dt in enumerate(dts):
        euler_data.append(-calculate_mean_position(read_data(euler_fns[i])[-1], B))
        cranic_data.append(-calculate_mean_position(read_data(cranic_fns[i])[-1], B))
        strang_data.append(-calculate_mean_position(read_data(strang_fns[i])[-1], B))
    # euler_data = euler_data[::-1]
    # cranic_data = cranic_data[::-1]
    # strang_data = strang_data[::-1]

    reference = strang_data[-1]

    euler_data = np.array(euler_data)
    cranic_data = np.array(cranic_data)
    strang_data = np.array(strang_data)
    euler_data =  np.abs(euler_data - strang_data)
    cranic_data = np.abs(cranic_data - strang_data)
    # strang_data = np.abs(strang_data - np.ones_like(strang_data) * reference)
    # euler_data = np.abs(euler_data - np.ones_like(strang_data) * reference)
    # cranic_data = np.abs(cranic_data - np.ones_like(strang_data) * reference)

    fig, axis = plt.subplots(figsize=(10,5))
    axis.grid()
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xlabel(r'$d\tau / \dfrac{\hbar}{E_0}$', fontsize=15)
    axis.set_ylabel(r'$|<x> - <x_{SS}>| / \Delta x$', fontsize=15)
    # axis.plot(dts_float[:-1],  euler_data[:-1], label='euler')
    # axis.plot(dts_float[:-1], cranic_data[:-1], label='canic')
    axis.plot(dts_float,  euler_data, label='euler')
    axis.plot(dts_float, cranic_data, label='canic')
    # axis.plot(dts_float[:-1], strang_data[:-1], label='strang')
    axis.invert_xaxis()
    fig.legend()
    plt.show()
