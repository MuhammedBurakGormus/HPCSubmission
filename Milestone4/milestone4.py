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
######

# Milestone 4: Couette Flow

def rigid_wall(f:np.ndarray,pos:str): 
    """
    This function updates&return the prob. density func. f applying a rigid wall bounday condition(i.e., bounce back)
    wrt to the position (pos) input. It basically matches the appropr. channels with their anti channels wrt to the position. 
    Notes:  index_channel_numbers = [0,1,2,3,4,5,6,7,8]
            index_anti_channel_numbers = [0,3,4,1,2,7,8,5,6]
            -----> ( f_updated(x_b,t+delta_t) = f(x_b,t) ) \n
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
            ----->  f_i_bar(x_b,t+delta_t) = f_i_star(x_b,t)  - 2 * w_i * rho_w  *  (c_i . u_w / c_s^2) \n
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
    c = milestone1.get_velocity_set()

    # Call the weights for D2Q9 lattice
    wi = milestone2.get_weights()

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
            u = milestone1.calculate_velocity_field(f_time[time,:,:,:])
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

if __name__ == '__main__':

    # Set the parameters for the test

    # Set the geometry of the domain
    width = 80
    height = 60   

    # Set the relaxation parameter
    w = 1.2

    # Set the velocity vector for the wall
    u_wall = np.array([0.1, 0]) # First argument is the velocity in the x direction, second is for y direction.

    # Set the time (at t=0; initial calculated pdf)
    time_steps = 15000
    plot_time_steps = 1000

    # Set the output folder to save the pictures there
    output_loc = os.getcwd() + "/Milestone4/" 

    ################ roadmap number 1 ######################
    # set initial values for density and velocity 
    rho_initial =  np.ones((width,height))
    u_initial = np.zeros((2,width,height))
    # calculate the starting prob. density func.
    f = milestone2.calculate_f_equilibrium(rho_initial,u_initial)

    # Create f_time to store the pdfs over time
    f_time = np.zeros((time_steps+1,9,width,height))
    f_time[0,:,:,:] = f

    # Loop for the Coutte flow following the sequence below
    for time_step in range(0,time_steps):
        print(time_step)
        # Starting with collision step 
        f = milestone2.collision(f,w)
        # Streaming
        f = milestone1.streaming(f)
        # Update the BC - Rigid Wall at the bottom
        f = rigid_wall(f,"bottom")
        # Update the BC - Moving Wall at the top
        f = moving_wall(f,u_wall,"top")
        # also save the prob. density functions values over time steps
        f_time[time_step+1,:,:,:] = f

    ## check mass conservation in the end ## 
    pdf_initial = f_time[0,:,:,:]
    pdf_last = f_time[time_steps,:,:,:]
    milestone1.check_mass_conservation(pdf_initial,pdf_last,"Couette Flow")

    # Line Plot of the velocity in the x direction at x = width/2
    output_name = f"velocity_wrt_y_width_{width:.2f}_height_{height:.2f}_w_{w:.2f}"
    plot_x_velocity_wrt_y(f_time,plot_time_steps,w,u_wall,output_name=output_name,output_loc=output_loc)

    #stream plot of the final state
    velocity_field = milestone1.calculate_velocity_field(pdf_last)
    output_name = f"2D_velocity_Couette_flow_width_{width:.2f}_height_{height:.2f}_w_{w:.2f}"
    title = f"Velocity profile x; t={time_steps}"
    milestone1.visualize_velocity_field_x(velocity_field,title=title,add_wall=True,pos=["Bottom","Top"],output_name=output_name,output_loc=output_loc) 


