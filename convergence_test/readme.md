# Test if Euler, Crank-Nicolson and Strang-Splitting Converge to the same values for suitable small time steps
To test if the Euler-, Crank-Nicolson- and Strang-Splitting integrators yield the same expectation values
for suitable small timesteps we calculate the difference of the expectation value of the position
after a classical forbidden tunneling process through a barrier.
## Usage 

- Choose the timestep sizes
- Generate the dipf-files accordingly
- Run the simulation and store the results in [eu, cn, ss]_{dt}.npy
- Write the timesteps in the array dts of convergence_plot.py
- Invoke convergence_plot.py with python -m tdse.convergence_plot $B where B is the barrier width used in the simulation
