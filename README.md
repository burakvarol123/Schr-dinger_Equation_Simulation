# Numerical Simulation of the Schrödinger Equation

In the following project, you can find the Numerical Simulation programs for simulating 
Time independent and Time dependent Schrödinger Equation. The observables we calculated and the details of physical parts are given in the related reports that you can find in the
repository. This project is done by Burak Varol, Lyding Brumm and Max Heimann.

# Usage
## Time independent Schrödinger Equation(tise):

 After you download the python scripts, you can create your own parameter file (.dipf), which has the template: 

        'eps': float,
        'mu': float,
        'N': int,
        'D': int,
        'tolerance_cg': float,
        'tolerance_pm': float,
        'res_recal_iterations': int,
Using the main function as 

        python -m tise.main $path_to_dipf_file 
will calculate the observables and write them back to the parameter file. You can find related examples in our data in the repository. 
To run the tests, please run with:
                 
        python -m tise.<modulename>
since the testes are implemented in every module in itself.
## Time Dependent Schrödinger Equation:
You must use parameter file with following template:
        
        'B': int,
        'mu': float,
        'N': int,
        'D': int,
        'tolerance_cg': float,
        'res_recal_iterations': int,
        's': float,
        'k': lambda s: np.fromstring(s.strip('[').strip(']'),\
                dtype=float, sep=','),
        'n0': lambda s: np.fromstring(s.strip('[').strip(']'),\
                dtype=float, sep=','),
        'dt': float,
        'T': float,
        'gen_propagator': lambda s: {
            'euler': gen_euler_propoagator,
            'strang_splitting': gen_strangsplitting_propagator,
            'cranic': gen_cranic_propagator}[s],
details for the related parameters can be found in the related report. You can choose between different propagators. To run, please use:

Simulate the evolution of the wave packet: 
      
    python -m tdse.main $path_to_dipf_file

Analyse:  
        
     python -m tdse.analyse $path_to_dipf_file
