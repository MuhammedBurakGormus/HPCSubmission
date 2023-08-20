In this folder, you can find the following files: 
* Final report for the class as a pdf named "Report_Burak_Görmüs.pdf"
* Report files that are used to generate the final report (i.e., pictures, .bib, and .tex files)
* **5 .py files** that are used to generate the final results. **(EXPLANATIONS ON HOW TO USE THEM IS BELOW.)** These files correspond to the simulations: 
    * Shear wave decay *(Milestone3)* --> shearwave_decay.py
    * Couette flow *(Milestone4)* --> couette_flow.py
    * Poiseuille flow *(Milestone5)* --> poiseuille_flow.py
    * Sliding lid serial *(Milestone6)* --> serial_sliding_lid.py
    * Sliding lid parallel *(Milestone7)* --> parallelized_sliding_lid.py
* Milestone folders from 1 to 7. In this folders, you can find corresponding .py files only containing the functions related to the specific milestone. All other necessary functions are called from other milestone folders. Besides, you can find additional pictures that are saved during running different experiments. 

---
**HOW TO USE THE SCRIPTS:**

In general, when a script *(shearwave_decay.py,couette_flow.py,poiseuille_flow.py, and serial_sliding_lid.py, and parallelized_sliding_lid.py)* is run, the script creates a folder in the current directory named as "GeneratedResults" (if it does not already exist), and saves the output pictures inside of this folder. 

In the beginning of each script, the parameters that are needed to be given to run the simulation are given. For example, for *couette_flow.py*, the first 15 lines are given as: 
```python
import sys
import os
############### CHANGE THE PARAMETERS BELOW (IF NEEDED) AND RUN THE SCRIPT ####################
# Set the geometry of the domain
width = 80
height = 60   
# Set the relaxation parameter
w = 1.2
# set the velocity in the x direction for the wall
u_wall_x = 0.3
# Set the time (at t=0; initial calculated pdf)
time_steps = 15000
plot_time_steps = 1000
################################################################################################
################################################################################################

# SOME FUNCTIONS HERE # 

################################################################################################
################################################################################################

if __name__ == '__main__': # Line 826 
    couette_flow(width,height,w,u_wall_x,time_steps,plot_time_steps)
```
The variables in the beginning of each script are the required parameters to run the function for the script. 

To run a specific simulation, just simply change the parameters in the beginning and run the code.

Or simply call the functions that are named the same as the script files. Since the functions are documented, the user can understand what parameters that needed to be given to run the simulation. 

--- 
**NOTE:**

For the parallel code *parallelized_sliding_lid.py*, the code is as follows: 
```python
import sys
import os
from mpi4py import MPI
############### CHANGE THE PARAMETERS BELOW (IF NEEDED) AND RUN THE SCRIPT ####################
############## (USING LOCAL TERMINAL) mpirun -n NxM python3 parallelized_sliding_lid.py ##########
# NxM is an integer that should be written down to describe the number of processors to be used, if the local terminal is used. 
# Set the geometric parameters
width = 75
height = 75
# Set kinematic viscosity
v = 0.03
# set the velocity in the x direction for the wall
u_wall_x = 0.1
# Set the division parameters of the domain for parallelization
N = 2 # set the number of subdomains in the x direction 
M = 2 # set the number of subdomains in the y direction 
# Set the time
time_steps = 1000
plot_time_steps = 500
################################################################################################
################################################################################################

# SOME FUNCTIONS HERE # 

################################################################################################
################################################################################################
if __name__ == '__main__': # Line 700
    parallelized_sliding_lid(width,height,N,M,v,u_wall_x,time_steps,plot_time_steps)
```
To run the code *(parallelization on the local computer)*, simply enter the command: 

**mpirun -n NxM python3**

where N and M stand for the #processes in the x and y direction. To be able to run the code, the values of N&M that are given in the beginning of the script should be compatible with the command above.

! While running the code on BwUniCluster, this is not important since the cloud determines the required processes automatically. ! 

Regarding the computations on BwUniCluster, an example example.job file to run the code on remote is as follows: 
```
#!/bin/bash -x
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:40:00
#SBATCH -J HPC_WITH_PYTHON
#SBATCH --mem=6gb
#SBATCH --export=ALL
#SBATCH --partition=multiple

module load devel/python/3.8.6_gnu_10.2
module load compiler/gnu/10.2
module load mpi/openmpi/4.1

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."
time mpirun python parallelized_sliding_lid.py
```
After saving the parallelized_sliding_lid.py and example.job at the same directory, running the command: 
```
sbatch example.job
```
is sufficient to run the code on parallel. 
**Just simply change the parameters N and M in the beginning of the parallelized_sliding_lid.py script.**

--- 
**NOTE:**

To save the generated pictures inside another folder rather than generated *GeneratedResults* folder, go to the script and create a **output_loc** parameter specifying the desired path to save the pictures, and give this variable as an input to the mentioned functions:
- shearwave_decay(....,output_loc=output_loc), 
- couette_flow(....,output_loc=output_loc),
- poiseuille_flow(....,output_loc=output_loc),
- serial_sliding_lid.py(....,output_loc=output_loc),
- parallelized_sliding_lid(....,output_loc=output_loc)

--- 
**NOTE:**

While running the *shearwave_decay.py*, the user is asked to which test to be performed. The user is required to enter either 1 or 2 to let the code continue running.
- If the user enters 1, the study for the sinusoidal density disturption is done. 
- If the user enters 2, the study for the sinusoidal velocity disturption is done. 

*The information related to this initialization can be understand better after reading the section 4.1 on the report. Simply, when choosing 1, the user initializes the simulation with Eq. 4.1, and when choosing 2, the user initializes the simulation with Eq. 4.2.*

--- 
**EXTRA NOTE:**

If you want to run any script/function inside the folders named as Milestonex, please make sure that before running the script the current directory is set to HPC_Submission path (the path of this main folder) because the scripts there save the pictures inside the folders depending on the their path unlike generating a folder named as GeneratedResults that is done by *(shearwave_decay.py,couette_flow.py,poiseuille_flow.py, and serial_sliding_lid.py, and parallelized_sliding_lid.py)*.  If any following script *(shearwave_decay.py,couette_flow.py,poiseuille_flow.py, and serial_sliding_lid.py, and parallelized_sliding_lid.py)* is run, this is not the case. 

--- 
**NOTE:**

The locations of the pictures that are saved after running a simulation is also printed out at the terminal. To see their directory, just check to terminal history. 





    
  


