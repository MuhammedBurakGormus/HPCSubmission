import sys
import os
############### CHANGE THE PARAMETERS BELOW (IF NEEDED) AND RUN THE SCRIPT ####################
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
################################################################################################
################################################################################################
################################################################################################
#################### BELOW ALL THE FUNCTIONS THAT ARE NEEDED FOR THIS SCRIPT ###################
############################### COMING FROM PREVIOUS MILESTONES ################################
#######
## import general modules
import math 
import numpy as np 
import matplotlib.pyplot as plt
######
# Functions for Poiseuille Flow
def poiseuille_flow(width:int,height:int,w:float,p_in:float,p_out:float,time_steps:int,plot_time_steps:int,output_loc:str=None): 
    """
    This function simulates Poiseuille flow, and returns the final probability density function. 
    Also, it plots&saves the velocity in the middle of the channel at x=width/2 in each plot_time_steps, and returns the line plot of the velocity
    evolution as well as the analytical solution to the problem. 
    Besides, it plots&saves 2D velocity profiles in the x direction in each plot_time_steps.
    Lastly, it plots&saves the density profile along the x direction at y = height/2. 
    Args: 
        width: int 
            The length of the domain in the x direction. 
        height: int 
            The length of the domain in the y direction. 
        w: float 
            Relaxation parameter to be used in the simulation.
        p_in: float
            Inlet pressure.
        p_out: float
            Outlet pressure. 
        time_steps: int
            The total time to run the simulation. 
        plot_time_steps: int
            In each plot_time_steps, the streamplot is created & saved. 
        output_loc: str
            The pictures are saved at os.getcwd() + output_loc. If it is not given, pictures are saved in the current directory in GeneratedResults folder that is created.
    Returns:
        f: np.ndarray
            The final probability density function. 
    IF OUTPUT LOCATION (output_loc) IS NOT GIVEN TO THE FUNCTION, THE PLOTS ARE SAVED IN THE GeneratedResults folder CREATED IN THE CURRENT DIRECTORY. 
    """
    # Set the directory
    if output_loc is None: 
        folder_name = "GeneratedResults"
        # Check if the folder exists
        if not os.path.exists(folder_name):
            # Create the folder
            os.mkdir(folder_name)
            print(f"Folder '{folder_name}' created. Output pictures are put inside this folder.")
        else:
            print(f"Folder '{folder_name}' already exists. Output pictures are inside this folder.")
        output_loc = os.getcwd() + "/GeneratedResults/"

    # Create the initial density and velocity
    rho_initial =  np.ones((width+2,height))
    u_initial = np.zeros((2,width+2,height))
    grid = calculate_f_equilibrium(rho_initial,u_initial) # grid is the extended pdf in x direction (have a size of (9,width+2,height))

    # Calculate inlet and outlet density arrays that are constants throughout the simulation
    rho_in = inlet_density_calculate(p_in,p_out) * np.ones((1,height)) # the size is 1 x height 
    rho_out = outlet_density_calculate(p_out) * np.ones((1,height)) # the size is 1 x height 

    # Also save the prob. density functions values over time steps 
    f_time = np.zeros((time_steps+1,9,rho_initial.shape[0],rho_initial.shape[1]))
    f_time[0,:,:,:] = grid
    virtual_inlet_equilibrium, virtual_outlet_equilibrium = calculate_equilibrium_for_virtual_nodes(grid,rho_in,rho_out)

    for time_step in range(0,time_steps): 
        if time_step % plot_time_steps == 0:
            print("Time step:",time_step)
        # streaming 
        grid = streaming(grid)
        # Update the BC - Rigid Wall at the bottom
        grid = rigid_wall(grid,"bottom")
        # Update the BC - Moving Wall at the top
        grid = rigid_wall(grid,"top")
        # after streaming update the grid applying periodic pressure BCs 
        grid = periodic_bc_update(grid,rho_in,rho_out,virtual_inlet_equilibrium, virtual_outlet_equilibrium)
        # perform collision
        grid = collision(grid,w)
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
        velocity_field = calculate_velocity_field(f)
        # Get the density
        rho = calculate_density(f)
        # Visualize density
        #title = f"Density profile at time:{time}"
        #output_name = f"density_profile_2D_pin_{p_in}_pout_{p_out}_time_{time}_width_{width}_height_{height}_w_{w}"
        #milestone1.visualize_density(rho,title=title,output_name=output_name,output_loc=output_loc)
        # Visualize velocity field
        title = f"Velocity in x direction at time:{time}"
        output_name = f"velocity_profile_2D_pin_{p_in}_pout_{p_out}_time_{time}_width_{width}_height_{height}_w_{w}"
        visualize_velocity_field_x(velocity_field,title=title,output_name=output_name,output_loc=output_loc) 

    # Visualize the velocity field lineplot for each plot_time_steps
    output_name= f"velocity_wrt_y_pin_{p_in}_pout_{p_out}_time_{time}_width_{width}_height_{height}_w_{w}"
    result_plot = plot_x_velocity_wrt_y(f_time,plot_time_steps,w=w,output_name=output_name,output_loc=output_loc)

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
    
    return f_time[-1,:,:,:] # final pdf

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
    u = calculate_velocity_field(grid)

    # Obtain the velocities at x = 1 & x = N
    u_1 = u[:,1,:]
    u_1 = np.reshape(u_1,(u_1.shape[0],1,u_1.shape[1])) # match the size for the function calculate_f_equilibrium
    u_N = u[:,-2,:]
    u_N = np.reshape(u_N,(u_N.shape[0],1,u_N.shape[1])) # match the size for the function calculate_f_equilibrium
    
    # Calculate the equilibrium given inlet density and u(x=N)
    virtual_inlet_equilibrium = calculate_f_equilibrium(rho_in,u_N)
    virtual_inlet_equilibrium = np.reshape(virtual_inlet_equilibrium,(virtual_inlet_equilibrium.shape[0],virtual_inlet_equilibrium.shape[2]))

    # Calculate the equilibrium given outlet density and u(x=1)
    virtual_outlet_equilibrium = calculate_f_equilibrium(rho_out,u_1) 
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
    rho = calculate_density(grid)
    u = calculate_velocity_field(grid)
    f_equilibrium = calculate_f_equilibrium(rho,u)

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
    rho = calculate_density(f)
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
######################## FUNCTIONS FROM PREVIUOS MILESTONES ############################
def get_velocity_set():
    """
    This function returns the velocity set for the simulations. \n (FROM MILESTONE1)
    Returns: 
        c: np.ndarray 
            The velocity set stated on the Milestone 1 that has a size of 2 x 9
    """
    # Create the velocity set c
    c  = np.transpose(np.array([[0,1,0,-1,0,1,-1,-1,1],[0,0,1,0,-1,1,1,-1,-1]]))
    return c

def get_weights(): 
    """
    This function returns the weights for the prob. dens. func. for the simulations.  (FROM MILESTONE2)
    Returns: 
        wi: np.ndarray 
            The weights stated on the Milestone 2 that has a size of 1 x 9
    """
    wi =  np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
    return wi

def streaming(f:np.ndarray):
    """
    This function calculates & returns the probability density f after streaming step using np.roll. \n (FROM MILESTONE1)
    Args: 
        f: np.ndarray
            The probability density function that has a size of 9 x width x height
    Returns: 
        f: np.ndarray
            The probability density function that has a size of 9 x width x height after the streaming operation. 
    """ 
    # Call the velocity set
    c = get_velocity_set()
    # Rolling the pdf over the 9 channels 
    for i in np.arange(1,9): 
        f[i,:,:] = np.roll(f[i,:,:] ,shift=c[i],axis=[0,1])
    return f

def calculate_density(f:np.ndarray):
    """
    This functions sums up all of the particles in the 9 channels for a given prob. density funct. f 
    along axis 0, and returns the density. \n (FROM MILESTONE1)
    Args:
        f: np.ndarray
            The probability density function that has a size of 9 x width x height
    Returns:     
        rho: np.ndarray
            The density that has a size of width x height
    """
    rho = np.sum(f,axis=0)
    return rho

def calculate_velocity_field(f:np.ndarray):
    """
    This function calculates & returns the velocity field for a given prob. density funct. f.\n (FROM MILESTONE1)
    Args: 
        f: np.ndarray
            The probability density function that has a size of 9 x width x height
    Returns: 
        u: np.ndarray
            The velocity field that has a size of 2 x width x height. 
    """
    # Construct velocity field of zeros v that has a size of 2 x width x height wrt to the given prob. density funct. f.
    u = np.zeros((2,np.size(f,axis=1),np.size(f,axis=2)))
    # Call the velocity set
    c = get_velocity_set()
    # Summing along axis 0
    integral = np.einsum("pxy,pk -> kxy", f,c)
    # Calculating the density for a given pdf.
    rho = calculate_density(f)
    # Define epsilon preventing zero division in any case
    epsilon = 1e-10  # or any other small value
    # Calculate the velocity field
    u = integral/(rho[None,:,:]+epsilon)
    return u

def calculate_f_equilibrium(rho:np.ndarray, u:np.ndarray): 
    """
    This function returns prob. density funct. for equilibrium feq that has a size of 9 x width x height. (FROM MILESTONE2)
    Args:
        rho: np.ndarray
            The density that has a size of width x height
        u: np.ndarray
            The velocity field that has a size of 2 x width x height. 
    Returns: 
        feq: np.ndarray
            The equilibrium prob. density function.
    """
    # Create empty feq to be returned 
    width = rho.shape[0]
    height = rho.shape[1]
    feq = np.zeros((9,width,height))

    # Call the velocity set
    c = get_velocity_set()

    # Call the weights for D2Q9 lattice
    wi = get_weights()

    # loop over 9 channels           
    for i in range(0,9):
        # Expand dimensions of c[i] to match the shape of u
        expanded_c_i = np.expand_dims(c[i], axis=(1, 2))    
        # Compute the dot product between expanded_c_i and u
        dot_product = np.sum(expanded_c_i * u, axis=0)  # shape: (15, 10)
        # Compute the square of the dot product between expanded_c_i and u
        dot_product_square = dot_product * dot_product
        # Compute the square of velocity 
        u_square = np.einsum("kxy,kxy->xy",u,u)
        # Calculate rhs
        rhs = 1 + 3 * dot_product + 9/2 * dot_product_square - 3/2 * u_square
        # Multiply the rhs with rho
        rho_rhs = rho * rhs 
        # Multiply the result with wi[i]
        result = wi[i] * rho_rhs
        feq[i,:,:] = result
    return feq

def collision(f:np.ndarray,w:float):
    """
    This function returns prob. density funct. after collision using Milestone 2 equation 2. \n (FROM MILESTONE2)
    Args:
        f: np.ndarray
            The prob. density function before the collision that has a size of 9 x width x height
        w: float
            The relaxation parameter that should have a value between 0 and 2.
    Returns: 
        f_collision: np.ndarray
             Prob. density function after collision that has a size of 9 x width x height
    """
    # Calculate the density before collision
    rho = calculate_density(f)
    # Calculate the velocity field before collision
    u = calculate_velocity_field(f)
    # Calculate the equilibrium prob. density funct. 
    feq = calculate_f_equilibrium(rho,u)
    # Obtain the pdf. after collision
    f_collision = f + w * (feq-f) 
    return f_collision

def rigid_wall(f:np.ndarray,pos:str): 
    """
    This function updates&return the prob. density func. f applying a rigid wall bounday condition(i.e., bounce back)
    wrt to the position (pos) input. It basically matches the appropr. channels with their anti channels wrt to the position. 
    Notes:  index_channel_numbers = [0,1,2,3,4,5,6,7,8]
            index_anti_channel_numbers = [0,3,4,1,2,7,8,5,6]
            -----> ( f_updated(x_b,t+delta_t) = f(x_b,t) ) \n (FROM MILESTONE4)
    Args: 
        f: np.ndarray
            The prob. density function before applying the BC that has a size of 9 x width x height
        pos: str 
            either top, bottom, left, or right specifying the position of the wall
    Returns: 
        f_updated: np.ndarray 
            The prob. density function after applying the BC that has a size of 9 x width x height
    """
    # Obtain the width and height from the prob. density funct. 
    width = f.shape[1] - 1 # -1 bcs indexing starts from 0
    height = f.shape[2] -1 # -1 bcs indexing starts from 0

    # Create a copy of f to update it without updating the f itself
    f_updated= f.copy()

    # Adding conditional statements based on the position of the wall
    if pos == "bottom": 
        # This cond. will effect the nodes at y = 0
        f_updated[2,:,0] = f[4,:,0] 
        f_updated[5,:,0] = f[7,:,0] 
        f_updated[6,:,0] = f[8,:,0] 
    elif pos == "top": 
        # This cond. will effect the nodes at y = height
        f_updated[4,:,height] = f[2,:,height] 
        f_updated[7,:,height] = f[5,:,height] 
        f_updated[8,:,height] = f[6,:,height] 
    elif pos == "left": 
        # This cond. will effect the nodes at x = 0 
        f_updated[1,0,:] = f[3,0,:] 
        f_updated[5,0,:] = f[7,0,:] 
        f_updated[8,0,:] = f[6,0,:] 
    elif pos == "right": 
        # This cond. will effect the nodes at x = width
        f_updated[3,width,:] = f[1,width,:] 
        f_updated[7,width,:] = f[5,width,:]
        f_updated[6,width,:] = f[8,width,:] 
    return f_updated

def moving_wall(f:np.ndarray,u_wall:np.ndarray,pos:str):
    """
    This function updates&return the prob. density func. f applying a moving wall bounday condition wrt to 
    the wall velocity input u_wall and  the position (pos) input. It basically matches the appropr. channels with their anti channels.
    Notes:  index_channel_numbers = [0,1,2,3,4,5,6,7,8]
            index_anti_channel_numbers = [0,3,4,1,2,7,8,5,6]
            ----->  f_i_bar(x_b,t+delta_t) = f_i_star(x_b,t)  - 2 * w_i * rho_w  *  (c_i . u_w / c_s^2) \n (FROM MILESTONE4)
    Args: 
        f: np.ndarray
            The prob. density function before applying the BC that has a size of 9 x width x height
        u_wall: np.ndarray
            Velocity vector of the wall that has a size of 1x2  
            ---> [u_x,u_y]: First argument is the velocity in the x direction, second is for y direction.
        pos: str
            either top, bottom, left, or right specifying the position of the wall
    Returns: 
        f_updated: np.ndarray 
            The prob. density function after applying the BC that has a size of 9 x width x height
    """
    # Call the velocity set
    c = get_velocity_set()

    # Call the weights for D2Q9 lattice
    wi = get_weights()

    # Obtain the width and height from the prob. density funct. 
    width = f.shape[1] -1 # -1 bcs indexing starts from 0
    height = f.shape[2] -1 # -1 bcs indexing starts from 0

    # Create a copy of f to update it without updating the f itself
    f_updated= f.copy()

    # Adding conditional statements based on the position of the wall
    if pos == "bottom": 
        # This cond. will effect the nodes at y = 0
        f_updated[2,:,0] = f[4,:,0]  - ( 6 * wi[4] * f[:,:,0].sum() * (c[4] * u_wall/(width+1)).sum() )
        f_updated[5,:,0] = f[7,:,0]  - ( 6 * wi[7] * f[:,:,0].sum() * (c[7] * u_wall/(width+1)).sum() )
        f_updated[6,:,0] = f[8,:,0]  - ( 6 * wi[8] * f[:,:,0].sum() * (c[8] * u_wall/(width+1)).sum() )
    elif pos == "top": 
        # This cond. will effect the nodes at y = height
        f_updated[4,:,height] = f[2,:,height]  - ( 6 * wi[2] * f[:,:,height].sum() * (c[2] * u_wall/(width+1)).sum() )
        f_updated[7,:,height] = f[5,:,height]  - ( 6 * wi[5] * f[:,:,height].sum() * (c[5] * u_wall/(width+1)).sum() )
        f_updated[8,:,height] = f[6,:,height]  - ( 6 * wi[6] * f[:,:,height].sum() * (c[6] * u_wall/(width+1)).sum() )
    elif pos == "left": 
        # This cond. will effect the nodes at x = 0 
        f_updated[1,0,:] = f[3,0,:]  - ( 6 * wi[3] * f[:,0,:].sum() * (c[3] * u_wall/(height+1)).sum() )
        f_updated[5,0,:] = f[7,0,:]  - ( 6 * wi[7] * f[:,0,:].sum() * (c[7] * u_wall/(height+1)).sum() )
        f_updated[8,0,:] = f[6,0,:]  - ( 6 * wi[6] * f[:,0,:].sum() * (c[6] * u_wall/(height+1)).sum() )
    elif pos == "right": 
        # This cond. will effect the nodes at x = width
        f_updated[3,width,:] = f[1,width,:]  - ( 6 * wi[1] * f[:,width,:].sum() * (c[1] * u_wall/(height+1)).sum() )
        f_updated[7,width,:] = f[5,width,:]  - ( 6 * wi[5] * f[:,width,:].sum() * (c[5] * u_wall/(height+1)).sum() )
        f_updated[6,width,:] = f[8,width,:]  - ( 6 * wi[8] * f[:,width,:].sum() * (c[8] * u_wall/(height+1)).sum() )
    return f_updated

def relaxation_from_kinematic_viscosity(v:float):
    """
    This function calculates&returns the relaxation parameter w from kinematic viscosity.\n (FROM MILESTONE6)
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
    This function calculates&returns the Reynolds number using the formula L*u/v. (FROM MILESTONE6)
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
    using the formula 4 x width x height / 2 x (width + height). (FROM MILESTONE6)
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

def streamplot_velocity(velocity_field:np.ndarray,title:str="",output_name:str="output",show_colorbar:bool=True,output_loc:str=None,kinematic_vis:float=None,u_wall:float=None,Re:float=None,time:float=None):
    """
    This function plots the streamline for a given velocity field. \n
    Args:
        velocity_field: np.ndarray
            The velocity field that has a size of 2 x width x height
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        show_color_bar: bool
            If the color bar to be omitted, argument should be given as False.
        output_loc: str
            The location of the output file.
    Returns: 
        fig: Matplotlib figure 
    """
    # Set the figure size
    plt.figure(figsize=(16, 9))  

    # Set the font parameters
    font = { 'weight': 'normal',
            'size': 28} 
    # Set the font parameters for specific elements
    plt.rc('font', **font)
    plt.rc('axes', titlesize=28)  
    plt.rc('axes', labelsize=28)  
    plt.rc('xtick', labelsize=28)  
    plt.rc('ytick', labelsize=28)  
    plt.rc('legend', fontsize=18)  

    # Creating dataset
    x = np.arange(0, velocity_field.shape[1])
    y = np.arange(0, velocity_field.shape[2])
    
    # Creating grids
    X, Y = np.meshgrid(x, y)
    
    # x-component to the right
    u = np.transpose(velocity_field[0,:,:])
    
    # y-component zero
    v = np.transpose(velocity_field[1,:,:])
    
    fig = plt.figure(figsize=(16, 9))

    # Create a colormap
    colormap = plt.cm.get_cmap('viridis')

    # Calculate the velocity magnitude
    velocity_magnitude = np.sqrt(u**2 + v**2)

    # Plotting stream plot
    streamplot = plt.streamplot(X, Y, u, v,color=velocity_magnitude, cmap=colormap)

    # Set axis limits
    plt.xlim(0, velocity_field.shape[1] -1)
    plt.ylim(0, velocity_field.shape[2] -1)

    # set title
    plt.title(title)

    if show_colorbar == True:
        # Add a colorbar
        cbar = plt.colorbar(streamplot.lines, format='%.2f')
        cbar.set_label("|u|", labelpad=1)  # Adjust labelpad as nee

    # Put the annotation showing relaxation parameter / width / height / Wall velocity vector
    if time: 
        plt.text(0.95, 0.25, f"Time: {time} ", ha='right', va='top',fontweight="bold",color="red", fontsize=18,transform=plt.gca().transAxes)
    if u_wall: 
        plt.text(0.95, 0.1, f"Wall velocity: {u_wall} ", ha='right', va='top',fontweight="bold",color="red", fontsize=18,transform=plt.gca().transAxes)
    if kinematic_vis:
        plt.text(0.95, 0.15, f"Kinematic viscosity v: {kinematic_vis:.2f}", ha='right', va='top',fontweight="bold",color="red",fontsize=18, transform=plt.gca().transAxes)
    if Re: 
        plt.text(0.95, 0.2, f"Reynolds number: {Re:.2f}", ha='right', va='top',fontweight="bold",color="red", fontsize=18, transform=plt.gca().transAxes)
        plt.text(0.95, 0.05, f"Width: {velocity_field.shape[1]:.2f}, Height: {velocity_field.shape[2]:.2f}", ha='right', va='top',fontweight="bold",color="red" ,fontsize=18,transform=plt.gca().transAxes)
    
    # Set x and y labels
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')

    # Save the image 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone1/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name = output_loc + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")

    # Show the plot
    #plt.show()

    return fig

def plot_x_velocity_wrt_y(f_time:np.ndarray,plot_time_steps:int,w:float=None,u_wall:np.ndarray=None,title:str=None,output_name:str="output",output_loc:str=None): 
    """
    This function plots the velocity in the x direction wrt y at the middle of the channel (x=width/2) in each time_steps given. \n
    Args: 
        f_time: np.ndarray
            The prob. density function over the time that has a size of time x 9 x width x height
        plot_time_steps: int
            In each plot_time_steps, figure will be updated. 
        w: float
            Relaxation parameter used for the simulation to be used as an annotation.
        u_wall: np.ndarray
            Velocity of the wall used for the simulation to be used as an annotation.
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        output_loc: str
            The location of the output file.
    Returns: 
        plt object to modify it later
    """
    # Set the figure size
    plt.figure(figsize=(16, 9))  

    # Set the font parameters
    font = { 'weight': 'normal',
            'size': 28} 
    # Set the font parameters for specific elements
    plt.rc('font', **font)
    plt.rc('axes', titlesize=28)  
    plt.rc('axes', labelsize=28)  
    plt.rc('xtick', labelsize=28)  
    plt.rc('ytick', labelsize=28)  
    plt.rc('legend', fontsize=18)  
     
    # Get the width and height of the domain and time
    width = f_time.shape[2]
    height = f_time.shape[3] 
    timesteps = f_time.shape[0]

    # Get the x coordinate at x=width/2, and set the y coordinates.
    middle_width = int(width / 2) 
    y_coordinates = np.arange(0,height,1)

    for time in range(timesteps):
        if time % plot_time_steps == 0: 
            # Calculate the velocity
            u = calculate_velocity_field(f_time[time,:,:,:])
            x_velocities_at_middle = u[0,middle_width,:]
            plt.plot(y_coordinates,x_velocities_at_middle,label=f"Time Step: {time}",linewidth=3)
    
    if u_wall is not None:
        # Plot the straight line to represent wall at t = 0
        plt.plot([y_coordinates[-1],y_coordinates[-1]],[0,u_wall[0]],color="red",linewidth=5)

    plt.xlabel("Y coordinates")
    plt.ylabel("Velocities in the x direction at the middle")
    if title != None:
        plt.title(title)
    plt.legend()

    # Put the annotation showing relaxation parameter / width / height / Wall velocity vector
    if u_wall is not None: 
        plt.text(0.95, 0.2, f"Wall velocity: {u_wall} ", ha='right', va='top',fontweight="bold",fontsize=24, transform=plt.gca().transAxes)
    if w:
        plt.text(0.95, 0.15, f"Relaxation parameter ω: {w:.2f}", ha='right', va='top',fontweight="bold",fontsize=24, transform=plt.gca().transAxes)
    plt.text(0.95, 0.1, f"Width: {width:.2f}, Height: {height:.2f}", ha='right', va='top',fontweight="bold", fontsize=24,transform=plt.gca().transAxes)
    
    # Save the figure 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone4/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name = output_loc + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")

    return plt
    #plt.show()

def visualize_velocity_field_x(velocity_field:np.ndarray,title:str="",output_name:str="output",show_colorbar:bool=True,add_wall:bool=False,pos:list=None,output_loc:str=None):
    """
    This function plots the velocities in the x direction over the domain and saves it as .png in the same directory where the code is run. \n
    Args: 
        velocity_field: np.ndarray
            The velocity field that has a size of 2 x width x height
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        show_color_bar: bool
            If the color bar to be omitted, argument should be given as False.
        add_wall: bool
            If walls to be shown as lines, argument should be given as True.
        pos: list 
            The list of the wall positions. They can be "Bottom", "Top", "Left", "Right"
        output_loc: str
            The location of the output file.
    Returns: 
        tuple: fig, ax 
            Matplotlib figure and axes objects.
    """
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(16, 9))
    heatmap = ax.imshow(velocity_field[0,:,:].transpose())

    # Set the font parameters
    font = { 'weight': 'normal',
            'size': 28} 
    # Set the font parameters for specific elements
    plt.rc('font', **font)
    plt.rc('axes', titlesize=28)  
    plt.rc('axes', labelsize=28)  
    plt.rc('xtick', labelsize=28)  
    plt.rc('ytick', labelsize=28)  
    plt.rc('legend', fontsize=28)  

    # Set the title
    ax.set_title(title)

    # Add color bar
    if show_colorbar == True:
        cbar = plt.colorbar(heatmap)

    # Invert the y-axis
    ax.invert_yaxis()

    # Add walls as lines to the plot
    if add_wall:
        for wall_pos in pos: 
            # The walls are located halfway node distant (0.5) from the boundaries
            if wall_pos == "Bottom":
                ax.plot(np.arange(0,velocity_field.shape[1],1), np.zeros(velocity_field.shape[1])-0.5, color='red',linewidth=10)
            elif wall_pos == "Top":
                ax.plot(np.arange(0,velocity_field.shape[1],1), velocity_field.shape[2] * np.ones(velocity_field.shape[1]) - 0.5, color='red',linewidth=10)
    
    # Set x and y labels
    ax.set_xlabel('X coordinates')
    ax.set_ylabel('Y coordinates')

    # Save the image 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone1/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name =  output_loc + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")

    # Show the plot
    #plt.show()

    return fig, ax

  
def visualize_velocity_field_y(velocity_field:np.ndarray,title:str="",output_name:str="output",show_colorbar:bool=True,add_wall:bool=False,pos:list=None,output_loc:str=None):
    """
    This function plots the velocities in the y direction over the domain and saves it as .png in the same directory where the code is run. \n
    Args: 
        velocity_field: np.ndarray
            The velocity field that has a size of 2 x width x height
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        show_color_bar: bool
            If the color bar to be omitted, argument should be given as False.
        add_wall: bool
            If walls to be shown as lines, argument should be given as True.
        pos: list 
            The list of the wall positions. They can be "Bottom", "Top", "Left", "Right"
        output_loc: str
            The location of the output file.
    Returns: 
        tuple: fig, ax 
            Matplotlib figure and axes objects.
    """
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=(16, 9))
    heatmap = ax.imshow(velocity_field[1,:,:].transpose())

    # Invert the y-axis
    ax.invert_yaxis()

    # Set the title
    ax.set_title(title)

    # Add color bar
    if show_colorbar == True:
        cbar = plt.colorbar(heatmap)

    # Invert the y-axis
    ax.invert_yaxis()

    # Add walls as lines to the plot
    if add_wall:
        for wall_pos in pos: 
            # The walls are located halfway node distant (0.5) from the boundaries
            if wall_pos == "Bottom":
                ax.plot(np.arange(0,velocity_field.shape[1],1), np.zeros(velocity_field.shape[1])-0.5, color='red',linewidth=10)
            elif wall_pos == "Top":
                ax.plot(np.arange(0,velocity_field.shape[1],1), velocity_field.shape[2] * np.ones(velocity_field.shape[1]) - 0.5, color='red',linewidth=10)
    
    # Set x and y labels
    ax.set_xlabel('X coordinates')
    ax.set_ylabel('Y coordinates')

    # Save the image 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone1/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name)
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name = output_loc + output_name + ".png"
        plt.savefig(file_name)
        print(f"Picture saved at location: {file_name}")
    
    # Show the plot
    #plt.show()

    return fig, ax

if __name__ == '__main__':
     poiseuille_flow(width,height,w,p_in,p_out,time_steps,plot_time_steps)

    

    
    
