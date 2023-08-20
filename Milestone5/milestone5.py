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
######

# Milestone 5: Poiseuille Flow

def outlet_density_calculate(p_out:float):
    """
    This function calculates & returns the outlet density given the outlet pressure. \n
    Args:
        p_out: float
            Pressure value at the outlet
    Returns:
        rho_out: float
            Density value at the outlet
    """ 
    cs2 = 1/3
    rho_out = p_out / cs2
    return rho_out

def inlet_density_calculate(p_in:float,p_out:float): 
    """
    This function calculates & returns the inlet density given the inlet& outlet pressure. \n
    Args: 
        p_out: float
            Pressure value at the outlet
        p_in: float
            Pressure value at the inlet
    Returns:
        rho_in: float
            Density value at the inlet
    """
    cs2 = 1/3
    delta_p = p_in - p_out
    rho_in = (p_out + delta_p) /cs2
    return rho_in

def calculate_equilibrium_for_virtual_nodes(grid:np.ndarray,rho_in:np.ndarray,rho_out:np.ndarray):
    """
    This function calculates the equilibrium pdf for the virtual nodes before collision at x = 0 and x= N+1 
    using rho_in, rho_out and velocity fields u_1, u_N at x = 1 and x = N. 
    (Calculating the first term at the right side of the equation.)\n
    Args: 
        grid: np.ndarray
            Probability density function that has extra layer of nodes outside having a size of 9 x width+2 x height
        rho_in : np.ndarray
            Inlet Density having a size of 1 x height
        rho_out : np.ndarray
            Outlet Density having a size of 1 x height
    Returns:
        virtual_inlet_equilibrium, virtual_outlet_equilibrium: tuple 
            Prob. density function for the virtual nodes. Each element in tuple is np.ndarray.
    """
    # Calculate the velocity field u for the grid
    u = milestone1.calculate_velocity_field(grid)

    # Obtain the velocities at x = 1 & x = N
    u_1 = u[:,1,:]
    u_1 = np.reshape(u_1,(u_1.shape[0],1,u_1.shape[1])) # match the size for the function calculate_f_equilibrium
    u_N = u[:,-2,:]
    u_N = np.reshape(u_N,(u_N.shape[0],1,u_N.shape[1])) # match the size for the function calculate_f_equilibrium
    
    # Calculate the equilibrium given inlet density and u(x=N)
    virtual_inlet_equilibrium = milestone2.calculate_f_equilibrium(rho_in,u_N)
    virtual_inlet_equilibrium = np.reshape(virtual_inlet_equilibrium,(virtual_inlet_equilibrium.shape[0],virtual_inlet_equilibrium.shape[2]))

    # Calculate the equilibrium given outlet density and u(x=1)
    virtual_outlet_equilibrium = milestone2.calculate_f_equilibrium(rho_out,u_1) 
    virtual_outlet_equilibrium = np.reshape(virtual_outlet_equilibrium,(virtual_outlet_equilibrium.shape[0],virtual_outlet_equilibrium.shape[2]))

    return virtual_inlet_equilibrium, virtual_outlet_equilibrium

def periodic_bc_update(grid:np.ndarray,rho_in:np.ndarray,rho_out:np.ndarray,virtual_inlet_equilibrium:np.ndarray, virtual_outlet_equilibrium:np.ndarray):
    """
    This function calculates the overall pdf for the virtual nodes at x = 0 and x = N+1. It means it updates&returns the grid.\n
    Args: 
        grid: np.ndarray
            Probability density function that has extra layer of nodes outside having a size of 9 x width+2 x height
        rho_in : np.ndarray
            Inlet Density having a size of 1 x height
        rho_out : np.ndarray
            Outlet Density having a size of 1 x height
        virtual_inlet_equilibrium:
            Equilibrium pdf for the virtual nodes before collision at x = 0 calculated via calculate_equilibrium_for_virtual_nodes
        virtual_outlet_equilibrium:
            Equilibrium pdf for the virtual nodes before collision at x = N+1 calculated via calculate_equilibrium_for_virtual_nodes
    Returns: 
        grid: np.ndarray
            Updated probability density function for the grid.
    """
    # Calculate the equilibrium pdf for x = 1 and x = N (last term at the right side of the equation)
    # Obtaining necessary terms for that
    rho = milestone1.calculate_density(grid)
    u = milestone1.calculate_velocity_field(grid)
    f_equilibrium = milestone2.calculate_f_equilibrium(rho,u)

    # get the equilibrium pdf for x = 1 
    f_x_1_eq = f_equilibrium[:,1,:]

    # get the equilibrium pdf for x = N 
    f_x_N_eq = f_equilibrium[:,-2,:]

    #calculate the non-equilibrium pdf for inlet
    virtual_inlet_non_equilibrium = grid[:,-2,:] - f_x_N_eq

    #calculate the non-equilibrium pdf for outlet 
    virtual_outlet_non_equilibrium = grid[:,1,:] - f_x_1_eq 

    #update the grid
    grid[:,0,:] = virtual_inlet_equilibrium + virtual_inlet_non_equilibrium
    grid[:,-1,:] = virtual_outlet_equilibrium + virtual_outlet_non_equilibrium

    return grid

def analytical_pouiseuille_flow(p_in:float,p_out:float,w:float,grid:np.ndarray):
    """
    This function returns the analytical solution for Pouiseuille flow.
    Args: 
        p_in: float
            Pressure value at the inlet
        p_out: float
            Pressure value at the outlet
        w: float
            Relaxation parameter. 
        grid: np.ndarray
            Probability density function that is used to calculate the density
    Returns: 
        u_y: np.ndarray
            Velocity in the x direction dependent of y coordinates
    """ 
    # get the geometric entities
    width = grid.shape[1] - 2 # excluding the virtual nodes
    height = grid.shape[2] 

    # calculate the dynamic_viscosity
    kinematic_viscosity = 1/3 * (1/w - 1/2)
    # Calculate average density
    rho = (grid[:,1:grid.shape[1]-1,:].sum()) / (width*height)
    dynamic_viscosity = rho * kinematic_viscosity

    # calculate pressure variation dp/dx
    dp_dx = (p_out - p_in) / width

    # for each node through the height calculate the velocity
    u_y = np.zeros((1,height))
    for y_coord in range(0,height,1):
        u_y_local = - (1/(2*dynamic_viscosity)) * dp_dx * y_coord * (height-1-y_coord)
        u_y[0,y_coord] = u_y_local

    return u_y

def plot_density_through_x(f:np.ndarray,output_name:str="output",output_loc:str=None):
    """
    This function plots the densities through the x direction at y = height/2. 
    Args:
        f: np.ndarray
            The prob. density function.
    Returns: 
        None. / JUST CAPTURE & SAVES THE LINEAR DENSITY PROFILE THROUGH THE CHANNEL.
    """
    # Set the figure size
    plt.figure(figsize=(16, 9))  
     
    # Get the width and height of the domain and time
    width = f.shape[1]
    height = f.shape[2] 

    # Get the x coordinate at x=width/2, and set the y coordinates.
    middle_height = int(height / 2) 
    x_coordinates = np.arange(0,width,1)

    # Calculate the density
    rho = milestone1.calculate_density(f)
    rho_at_middle = rho[:,middle_height]
    plt.plot(x_coordinates,rho_at_middle)
    plt.title("Density profile at the middle")
    plt.xlabel("X coordinates")
    plt.ylabel("Density")

    # Save the figure 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone5/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name)
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name = output_loc + output_name + ".png"
        plt.savefig(file_name)
        print(f"Picture saved at location: {file_name}")

    #plt.show()

if __name__ == '__main__':

    # Geometric entities of the domain
    width = 100
    height = 50

    # Pressure values at the inlet and outlet
    p_in = 1.005
    p_out = 1

    # Define time steps 
    time_steps = 10000
    plot_time_steps = 1000

    # Set the relaxation parameter
    w = 1

    # Set the output folder to save the pictures there
    output_loc = os.getcwd() + "/Milestone5/" 
    # Create the initial density and velocity
    rho_initial =  np.ones((width+2,height))
    u_initial = np.zeros((2,width+2,height))
    grid = milestone2.calculate_f_equilibrium(rho_initial,u_initial) # grid is the extended pdf in x direction (have a size of (9,width+2,height))

    # Calculate inlet and outlet density arrays that are constants throughout the simulation
    rho_in = inlet_density_calculate(p_in,p_out) * np.ones((1,height)) # the size is 1 x height 
    rho_out = outlet_density_calculate(p_out) * np.ones((1,height)) # the size is 1 x height 

    # Also save the prob. density functions values over time steps 
    f_time = np.zeros((time_steps+1,9,rho_initial.shape[0],rho_initial.shape[1]))
    f_time[0,:,:,:] = grid
    virtual_inlet_equilibrium, virtual_outlet_equilibrium = calculate_equilibrium_for_virtual_nodes(grid,rho_in,rho_out)

    for time_step in range(0,time_steps): 
        print(time_step)
        # streaming 
        grid = milestone1.streaming(grid)
        # Update the BC - Rigid Wall at the bottom
        grid = milestone4.rigid_wall(grid,"bottom")
        # Update the BC - Moving Wall at the top
        grid = milestone4.rigid_wall(grid,"top")
        # after streaming update the grid applying periodic pressure BCs 
        grid = periodic_bc_update(grid,rho_in,rho_out,virtual_inlet_equilibrium, virtual_outlet_equilibrium)
        # perform collision
        grid = milestone2.collision(grid,w)
        # before streaming calculate virtual equilibrium for virtual nodes
        virtual_inlet_equilibrium, virtual_outlet_equilibrium = calculate_equilibrium_for_virtual_nodes(grid,rho_in,rho_out)
        # also save the prob. density functions values over time steps
        f_time[time_step+1,:,:,:] = grid

    # Omit the extra nodes from f_time
    f_time = f_time[:,:,1:-1,:]

    # Visualize the velocity field & density in 2D for each plot_time_steps
    for time in range(0,time_steps+plot_time_steps,plot_time_steps):
        # Get the pdf for the time step
        f = f_time[time,:,:,:]
        # Get the velocity field
        velocity_field = milestone1.calculate_velocity_field(f)
        # Get the density
        rho = milestone1.calculate_density(f)
        # Visualize density
        #title = f"Density profile at time:{time}"
        #output_name = f"density_profile_2D_pin_{p_in}_pout_{p_out}_time_{time}_width_{width}_height_{height}_w_{w}"
        #milestone1.visualize_density(rho,title=title,output_name=output_name,output_loc=output_loc)
        # Visualize velocity field
        title = f"Velocity in x direction at time:{time}"
        output_name = f"velocity_profile_2D_pin_{p_in}_pout_{p_out}_time_{time}_width_{width}_height_{height}_w_{w}"
        milestone1.visualize_velocity_field_x(velocity_field,title=title,output_name=output_name,output_loc=output_loc) 

    # Visualize the velocity field lineplot for each plot_time_steps
    output_name= f"velocity_wrt_y_pin_{p_in}_pout_{p_out}_time_{time}_width_{width}_height_{height}_w_{w}"
    result_plot = milestone4.plot_x_velocity_wrt_y(f_time,plot_time_steps,w=w,output_name=output_name,output_loc=output_loc)

    # Scatter for Analytical Solution
    u_y_analytical = analytical_pouiseuille_flow(p_in,p_out,w,grid)
    u_y_analytical = np.reshape(u_y_analytical,(height,))
    y_coordinates = np.arange(0,height,1)
    result_plot.scatter(y_coordinates,u_y_analytical,label="Analytical",s=150)
    result_plot.legend()
    output_name= f"with_analytical_velocity_wrt_y_pin_{p_in}_pout_{p_out}_time_{time}_width_{width}_height_{height}_w_{w}" + ".png"
    plt.savefig(output_loc+output_name)

    # Plot the density along the centerline of the channel at t = time_steps
    output_name = "linear_density_profile"
    plot_density_through_x(f_time[-1,:,:,:],output_name=output_name,output_loc=output_loc)

