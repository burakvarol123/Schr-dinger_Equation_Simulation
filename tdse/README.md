# README #

This is a short into the usage of the tdse module from group poincare.
The scripts are intended to be run in the following way.

## creating a parameter.dipf file ##

- to read in the parameters you will need to specify teh parameters with which the code should run
- this should look like the following
'''
N=1024;
B=30;
mu=100.0;
D=1;
tolerance_cg=10e-10;
res_recal_iterations=100;
s=1;
k=[2.25];
n0=[-150];
dt=0.001;
T=2;
gen_propagator=strang_splitting;
out=pillow_example
'''
- save this file under name.dipf (you may of course choose another fileformat, but to keep track we always use .dipf)
- the explanation for each parameter may be extracted from the docstrings in main.py or the report

## running the code ##

- open a terminal
- cd to the folder cp-ii-problem-1
- type '''python3 -m tdse.script''' to run script.py
- in the case of modules which mainly are libraries of functions this will execute a test of the module
- to produce simulation data with parameters defined in name1.dipf, name2.dipf ... run main.py
  - '''python3 -m tdse.main path/to/file/name1.dipf path2/to2/file2/name2.dipf''
- This will simulate the propagation of the wavepacket according to the given parameters
- Now you can run analyse.py:
  - '''python3 -m tdse.main path/to/file/name1.dipf path2/to2/file2/name2.dipf''
- This will produce plots for all observables over time and a .gif animation of the wave propagating

## convergence ##

The convergence plot programe falls a little bit out of line, it is not realy meant to be exposed
to the user, but for sake of completeness the steps to produce a convergence plot are:

- Choose the timestep sizes
- Generate the dipf-files accordingly
- Run the simulation and store the results in [eu, cn, ss]_{dt}.npy
- Write the timesteps in the array dts of convergence_plot.py
- Invoke convergence_plot.py with python -m tdse.convergence_plot $B where B is the barrier width used in the simulation
