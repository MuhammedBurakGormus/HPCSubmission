#######
## import general modules
import math 
import numpy as np 
import matplotlib.pyplot as plt
import sys
import os
######

#######
## call the functions from other scripts
sys.path.append('../HPC_Submission/Milestone1')  # to call the functions from milestone1
import milestone1
sys.path.append('../HPC_Submission/Milestone2')  # to call the functions from milestone2
import milestone2
sys.path.append('../HPC_Submission/Milestone3')  # to call the functions from milestone3
import milestone3
sys.path.append('../HPC_Submission/Milestone4')  # to call the functions from milestone4
import milestone4
sys.path.append('../HPC_Submission/Milestone5')  # to call the functions from milestone4
import milestone5
######

# Milestone 6: The Sliding Lid

def relaxation_from_kinematic_viscosity(v:float):
    """
    This function calculates&returns the relaxation parameter w from kinematic viscosity.\n
    Args:
        v : float
            Kinematic viscocity
    Returns: 
        w : float
            Relaxation parameter
    """ 
    w = 1 / (3*v+0.5)
    return w

def calculate_reynolds_number(width:float,height:float,u:float,v:float):
    """
    This function calculates&returns the Reynolds number using the formula L*u/v
    Args: 
        width: float
            Width of the channel
        height: float
            Height of the channel
        u: float
            Flow velocity
        v: float
            Kinematic viscosity
    Returns:
        Re: float
            Reynolds number
    """
    L = calculate_hydraulic_diameter(width,height)
    Re = L * u / v
    return Re

def calculate_hydraulic_diameter(width:float,height:float): 
    """
    This function calculates&returns the hydraulic diameter as a characteristic length 
    using the formula 4 x width x height / 2 x (width + height)
    Args: 
        width: float
            Width of the channel
        height: float
            Height of the channel
    Returns:
        L: float
            Hydraulic diameter 
    """
    L = (4 * width * height) / (2 * (width + height))
    return L

if __name__ == '__main__':

    # Set the geometric parameters
    width = 75
    height = 75

    # Set kinematic viscosity
    v = 0.09

    # set the velocity wall
    u_wall = np.array([0.3, 0])

    # Set the time
    time_steps = 25000
    plot_time_steps = 500

    # Set the output folder to save the pictures there
    output_loc = os.getcwd() + "/Milestone6/" 

    # Print out the Reynolds number 
    Re = calculate_reynolds_number(width,height,u_wall[0],v)
    print(f"Reynolds number Re: {Re}")

    # set initial values for density and velocity 
    rho_initial =  np.ones((width,height))
    u_initial = np.zeros((2,width,height))

    # Obtain the starting prob. dens. funct. 
    f = milestone2.calculate_f_equilibrium(rho_initial,u_initial)

    # Calculate relaxation parameter
    w = relaxation_from_kinematic_viscosity(v)

    for time_step in range(0,time_steps):
        # Starting with collision step 
        f = milestone2.collision(f,w)
        # Streaming
        f = milestone1.streaming(f)
        # Update the BC - Rigid Wall at the bottom & left & right
        f = milestone4.rigid_wall(f,"bottom")
        f = milestone4.rigid_wall(f,"left")
        f = milestone4.rigid_wall(f,"right")
        # Update the BC - Moving Wall at the top
        f = milestone4.moving_wall(f,u_wall,"top")
        # Stream plot in each plot_time_steps
        if time_step % plot_time_steps == 0:
            print(f"time_step: {time_step}")
            velocity_field = milestone1.calculate_velocity_field(f)
            output_name = f"sliding_lid_u={u_wall[0]}_v={v}_width={width}_height={height}_time={time_step}_Re={Re}"
            milestone1.streamplot_velocity(velocity_field,output_name=output_name,output_loc=output_loc,kinematic_vis=v,u_wall=u_wall[0],Re=Re,time=time_step)
    
   