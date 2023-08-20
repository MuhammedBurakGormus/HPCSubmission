import sys
import os
############### CHANGE THE PARAMETERS BELOW (IF NEEDED) AND RUN THE SCRIPT ####################
# Set the geometric parameters
width = 75
height = 75
# Set kinematic viscosity
v = 0.03
# set the velocity in the x direction for the wall
u_wall_x = 0.1
# Set the time
time_steps = 5000
plot_time_steps = 500
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
def serial_sliding_lid(width:int,height:int,v:float,u_wall_x:float,time_steps:int,plot_time_steps:int,output_loc:str=None): 
    """
    This function simulates the sliding lid problem in serial, and returns the final probability density function. 
    Also, it plots the domain (streamplots) in plot_time_steps. 
    Args: 
        width: int 
            The length of the domain in the x direction. 
        height: int 
            The length of the domain in the y direction. 
        v: float 
            Kinematic viscosity to be used in the simulation.
        u_wall_x: float
            The speed of the top wall in the x direction. 
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

    # Print out the Reynolds number 
    Re = calculate_reynolds_number(width,height,u_wall_x,v)
    print(f"Reynolds number Re: {Re}")

    # set initial values for density and velocity 
    rho_initial =  np.ones((width,height))
    u_initial = np.zeros((2,width,height))

    # Obtain the starting prob. dens. funct. 
    f = calculate_f_equilibrium(rho_initial,u_initial)

    # Calculate relaxation parameter
    w = relaxation_from_kinematic_viscosity(v)

    # Set the u_wall vector 
    u_wall = np.array([u_wall_x, 0])

    # Set the directory
    if output_loc is None: 
        output_loc = os.getcwd() + "/"

    for time_step in range(0,time_steps):
        # Starting with collision step 
        f = collision(f,w)
        # Streaming
        f = streaming(f)
        # Update the BC - Rigid Wall at the bottom & left & right
        f = rigid_wall(f,"bottom")
        f = rigid_wall(f,"left")
        f = rigid_wall(f,"right")
        # Update the BC - Moving Wall at the top
        f = moving_wall(f,u_wall,"top")
        # Stream plot in each plot_time_steps
        if (time_step+1) % plot_time_steps == 0:
            print(f"time_step: {time_step+1}")
            velocity_field = calculate_velocity_field(f)
            output_name = f"sliding_lid_u={u_wall[0]}_v={v}_width={width}_height={height}_time={time_step+1}_Re={Re}"
            streamplot_velocity(velocity_field,output_name=output_name,output_loc=output_loc,kinematic_vis=v,u_wall=u_wall[0],Re=Re,time=time_step+1)
    
    return f 
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


##############################################################################################################################
##############################################################################################################################

if __name__ == '__main__':
    serial_sliding_lid(width,height,v,u_wall_x,time_steps,plot_time_steps)

    