#######
## import general modules
import math 
import numpy as np 
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
import os
######

# Milestone 7 - Parallelization using MPI

##############################################################################################################################
############################################# BELOW ARE THE FUNCTIONS ########################################################

def parallelization(width:int,height:int,N:int,M:int,v:float,u_wall:np.ndarray,time_steps:int,plot_time_steps:int):
    """
    The function solves the sliding lid problem in a parallized manner, plots&saves the stream plots in each plot_time_steps.
    Also, returns the final pdf.
    To run the code from the terminal, following command to be used:  (NxM processor)
        mpirun -n NxM python3 milestone7.py 
    Args:
        width: int
            Number of lattices in the x direction \n
        height: int
            Number of lattices in the y direction \n
        N: int
            Number of process to be used in the x direction \n
        M: int
            Number of  process to be used in the y direction \n
        u_wall: np.ndarray
            Velocity field of the top wall having a size of 1x2. First index is for the velocity in the x direction.
            Example; u_wall = np.ndarray([0.1,0])...
        time_steps: int
            Duration of the simulation
        plot_time_steps: int
            In each plot_time_steps, stream plots are saved. 
    Returns: 
        f_final: np.ndarray
            Final prob. density func. after combining all the process on rank 0. 
    """ 
    # Start Communication 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    # Create 2D commmunicator & get the neighbors for each process
    cartcomm = comm.Create_cart((N,M),periods = True)
    rcoords = cartcomm.Get_coords(rank)
    x_coord_rank = rcoords[0] # x coordinate of the rank
    y_coord_rank = rcoords[1] # y coordinate of the rank
    sR, dR = cartcomm.Shift(0,1)  # getting source&destination for the right direction
    sL, dL = cartcomm.Shift(0,-1) # getting source&destination for the left direction
    sU, dU = cartcomm.Shift(1,1)  # getting source&destination upwards
    sD, dD = cartcomm.Shift(1,-1) # getting source&destination downwards
  
    subdomain_size_in_x = width // N # size of a local domain in the x direction
    subdomain_size_in_y = height // M # size of a local domain in the y direction

    # Determine the x coordinates for the subdomains
    nx1 = x_coord_rank * subdomain_size_in_x # left boundary of the local domain 
    nx2 = (x_coord_rank+1) * subdomain_size_in_x# right boundary of the local domain

    # Determine the y coordinates for the subdomains
    ny1 = y_coord_rank * subdomain_size_in_y     # bottom of the local domain 
    ny2 = (y_coord_rank+1) * subdomain_size_in_y # top of the local domain

    # Adjustment of the upper row (if m=height/M is not a integer)
    if y_coord_rank == M-1: 
        ny2 = height
        subdomain_size_in_y = height - (M-1) * subdomain_size_in_y

    # Adjustment of the last column (if n=width/N is not a integer)
    if x_coord_rank == N-1: 
        nx2 = width
        subdomain_size_in_x = width - (N-1) * subdomain_size_in_x

    # Create the subdomains
    # Include the buffer nodes
    subdomain_x = np.arange(nx1-1,nx2+1,1)
     # Include the buffer nodes
    subdomain_y = np.arange(ny1-1,ny2+1,1)

    # Determining the BCs for the local subdomains [left,right,bottom,top]
    local_boundaries = assign_boundaries_to_subdomains(subdomain_x,subdomain_y,width,height)

    # set initial values for density and velocity for each subdomain; and creating the grid including the buffer nodes
    rho_initial =  np.ones((subdomain_size_in_x+2,subdomain_size_in_y+2))
    u_initial = np.zeros((2,subdomain_size_in_x+2,subdomain_size_in_y+2))
    grid = calculate_f_equilibrium(rho_initial,u_initial)
    #print(f"Rank: {rank} ---> Grid Size: {grid.shape}")

    # Calculate relaxation parameter
    w = relaxation_from_kinematic_viscosity(v)

    for time_step in range(0,time_steps):
        # Communicate Before Operations
        # Send to the right, receive from left 
        recvbuf = grid[:,0:1,:].copy()
        comm.Sendrecv(grid[:,-2:-1,:].copy(), dR, recvbuf=recvbuf,source=sR)
        grid[:,0:1,:] = recvbuf
        # Send to the left, receive from right
        recvbuf = grid[:,-1:,:].copy()
        comm.Sendrecv(grid[:,1:2,:].copy(), dL, recvbuf=recvbuf,source=sL)
        grid[:,-1:,:]= recvbuf
        # Send to the up, receive from down 
        recvbuf = grid[:,:,0:1].copy()
        comm.Sendrecv(grid[:,:,-2:-1].copy(), dU, recvbuf=recvbuf,source=sU)
        grid[:,:,0:1] = recvbuf
        # Send to the down, receive from top 
        recvbuf = grid[:,:,-1:].copy()
        comm.Sendrecv(grid[:,:,1:2].copy(), dD, recvbuf=recvbuf,source=sD)
        grid[:,:,-1:] = recvbuf

        # Starting with collision step 
        grid = collision(grid,w)
        # Streaming
        grid = streaming(grid)
        # Obtain the real domain to perform the BCs operations
        real_domain = obtain_real_domain(grid)
        #print(f"Rank: {rank} ---> Real Domain Size: {real_domain.shape}")
        #print()

        # Update the BC on the real domain based on the local_boundaries obtained above
        for bc in local_boundaries:
            if bc[0] != None: 
                bc_type = bc[0]
                bc_direction = bc[1]
                if bc_type == 'rigid_wall':
                    real_domain = rigid_wall(real_domain,bc_direction)
                if bc_type == 'moving_wall':
                    real_domain = moving_wall(real_domain,u_wall,bc_direction)

        # Modify the grid after the BCs operation
        grid = modify_grid(grid,real_domain)

        # In each plot_time_steps, results from different processes are combined & plotted & saved.
        if (time_step % plot_time_steps == 0) and ( time_step !=0 ):
            # Obtain the real domain for each process
            real_domain = obtain_real_domain(grid)
            # When n/N or m/M not integer, send the domain that has the shape of other ranks
            if real_domain.shape[1] != width // N:
                real_domain = real_domain[:,0:width//N,:]
            if real_domain.shape[2] != height // M:
                real_domain = real_domain[:,:,0:height//M]
            # Gather the data on whole_pdf without the extra nodes due to n/N or m/M ! = int
            whole_pdf = np.zeros((N,M,9,width // N,height//M))
            comm.Gather(real_domain.copy(),whole_pdf,root=0)

            # Gather the data coming from the extra nodes due to n/N or m/M ! = int
            # Get the extra columns
            whole_extra_columns = np.zeros((N,M,9,width%N,height//M))
            extra_columns = np.zeros((9,width%N,height//M))
            if rcoords[0] == N-1: 
                extra_columns =  obtain_real_domain(grid)[:,width//N:,:height//M]
            comm.Gather(extra_columns.copy(),whole_extra_columns,root=0)
            # Get the extra rows
            whole_extra_rows = np.zeros((N,M,9,width//N,height%M))
            extra_rows = np.zeros((9,width//N,height%M))
            if rcoords[1] == M-1: 
                extra_rows =  obtain_real_domain(grid)[:,:width//N,height//M:]
            comm.Gather(extra_rows.copy(),whole_extra_rows,root=0)
            # Get the corner info
            whole_extra_corners = np.zeros((N,M,9,width%N,height%M))
            extra_corners = np.zeros((9,width%N,height%M))
            if (rcoords[1] == M-1) and (rcoords[0] == N-1): 
                extra_corners =  obtain_real_domain(grid)[:,width//N:,height//M:]
            comm.Gather(extra_corners.copy(),whole_extra_corners,root=0)

            if rank == 0:

                # Save some pictures from rank 0 domain to show parallelization
                velocity_field_0 = calculate_velocity_field(real_domain)
                output_name_0 = f"rank_0_parallel_sliding_lid_u={u_wall[0]}_v={v}_width={width}_height={height}_time={time_step}_Re={Re}_N={N}_M={M}"
                streamplot_velocity(velocity_field_0,output_name=output_name_0,output_loc=output_loc,kinematic_vis=v,u_wall=u_wall[0],Re=Re,time=time_step)

                # Constructing the final pdf
                final_pdf = np.zeros((9,width,height))
                # Constructing the pdf without the extra nodes due to n/N or m/M ! = int
                last_pdf_without_extra_nodes = np.zeros((9,(width//N)*N,(height//M)*M))
                for grid_x_coord in range(whole_pdf.shape[0]): 
                    for grid_y_coord in range(whole_pdf.shape[1]): 
                        last_pdf_without_extra_nodes[:,grid_x_coord*(width//N):(grid_x_coord+1)*(width//N),grid_y_coord*(height//M):(grid_y_coord+1)*(height//M)] = whole_pdf[grid_x_coord,grid_y_coord,:,:,:]
                final_pdf[:,:(width//N)*N,:(height//M)*M] = last_pdf_without_extra_nodes

                # Adding the last columns info 
                if (width%N) != 0: 
                    for grid_x_coord in range(whole_extra_columns.shape[0]): 
                        for grid_y_coord in range(whole_extra_columns.shape[1]): 
                            if grid_x_coord == N-1:
                                final_pdf[:,(width//N)*N:,grid_y_coord*(height//M):(grid_y_coord+1)*(height//M)] = whole_extra_columns[grid_x_coord,grid_y_coord,:,:,:]
                
                # Adding the last rows info 
                if (height%M) != 0: 
                    for grid_x_coord in range(whole_extra_columns.shape[0]): 
                        for grid_y_coord in range(whole_extra_columns.shape[1]): 
                            if grid_y_coord == M-1:
                                final_pdf[:,grid_x_coord*(width//N):(grid_x_coord+1)*(width//N),(height//M)*M:] = whole_extra_rows[grid_x_coord,grid_y_coord,:,:,:]

                # Adding the corner info 
                if ((width%N) != 0) and ((height%M) != 0) : 
                    for grid_x_coord in range(whole_extra_columns.shape[0]): 
                        for grid_y_coord in range(whole_extra_columns.shape[1]): 
                            if (grid_y_coord == M-1) and (grid_x_coord == N-1):
                                final_pdf[:,(width//N)*N:,(height//M)*M:] = whole_extra_corners[grid_x_coord,grid_y_coord,:,:,:]
                
                # Result of combined processes
                print(f"Time step: {time_step}")
                velocity_field = calculate_velocity_field(final_pdf)
                output_name = f"parallel_sliding_lid_u={u_wall[0]}_v={v}_width={width}_height={height}_time={time_step}_Re={Re}_N={N}_M={M}"
                streamplot_velocity(velocity_field,output_name=output_name,output_loc=output_loc,kinematic_vis=v,u_wall=u_wall[0],Re=Re,time=time_step)
                
def assign_boundaries_to_subdomains(subdomain_x:np.ndarray,subdomain_y:np.ndarray,width:int,height:int):
    """
    This function returns a list of the BCs at the subdomain edges in the order [BC@left,BC@right,BC@bottom,BC@top] considering the sliding lid.\n
    subdomain_x:np.ndarray
        the array that contains the x coordinates of the real subdomain without buffer\n
    subdomain_y:np.ndarray
        the array that contains the y coordinates of the real subdomain without buffer\n
    width: int
        number of lattices in the x direction \n
    height: int
        number of lattices in the y direction \n
    """
    BCs = []
    # For the BC@left
    if subdomain_x[0] == -1:
        BCs.append(("rigid_wall","left"))
    else: 
        BCs.append((None,"left"))
    # For the BC@right
    if subdomain_x[-1] == width:
        BCs.append(("rigid_wall","right"))
    else: 
        BCs.append((None,"right"))
    # For the BC@bottom
    if subdomain_y[0] == -1:
        BCs.append(("rigid_wall","bottom"))
    else: 
        BCs.append((None,"bottom"))
    # For the BC@top
    if subdomain_y[-1] == height:
        BCs.append(("moving_wall","top"))
    else: 
        BCs.append((None,"top"))
    return BCs

def obtain_real_domain(grid:np.ndarray): 
    """
    This function returns the real domain excluding the buffer nodes. \n
    grid: np.ndarray
        pdf having a size of 9 x (width+2) x (height+2) including the buffer nodes in both directions x and y
    """
    return grid[:,1:-1,1:-1]

def modify_grid(grid:np.ndarray,f:np.ndarray): 
    """
    This function copies the start&last nodes of real domain and put them onto the grid, and returns the grid. \n
    grid: np.ndarray
        pdf having a size of 9 x (width+2) x (height+2) including the buffer nodes in both directions x and y. \n
    f: np.ndarray
        pdf having a size of 9 x width x height that is the real domain excluding buffer nodes
    """
    grid[:,1:-1,1:-1] = f
    return grid

############ BELOW ALL THE FUNCTIONS THAT ARE NEEDED FOR THIS SCRIPT ############
######################## COMING FROM PREVIOUS MILESTONES ########################

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
    This function plots the streamline for a given velocity field. \n (FROM MILESTONE1)
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

    # Put the annotation showing relaxation parameter / width / height / Wall velocity vector
    if time: 
        plt.text(0.95, 0.25, f"Time: {time} ", ha='right', va='top',fontweight="bold",color="red", transform=plt.gca().transAxes)
    if u_wall: 
        plt.text(0.95, 0.1, f"Wall velocity: {u_wall} ", ha='right', va='top',fontweight="bold",color="red", transform=plt.gca().transAxes)
    if kinematic_vis:
        plt.text(0.95, 0.15, f"Kinematic viscosity v: {kinematic_vis:.2f}", ha='right', va='top',fontweight="bold",color="red", transform=plt.gca().transAxes)
    if Re: 
        plt.text(0.95, 0.2, f"Reynolds number: {Re:.2f}", ha='right', va='top',fontweight="bold",color="red", transform=plt.gca().transAxes)
        plt.text(0.95, 0.05, f"Width: {velocity_field.shape[1]:.2f}, Height: {velocity_field.shape[2]:.2f}", ha='right', va='top',fontweight="bold",color="red", transform=plt.gca().transAxes)
    
    # Set x and y labels
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')

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

    return fig

##############################################################################################################################
##############################################################################################################################



##############################################################################################################################
############################################# BELOW CHANGE THE PARAMETERS AND ################################################
############################## RUN THE CODE VIA mpirun -n NxM python3 Milestone7/milestone7.py  ##############################

# Milestone 7 - Parallelization using MPI
# To run the code from the terminal, following command to be used:  (NxM processor)
# mpirun -n NxM python3 Milestone7/milestone7.py 

if __name__ == '__main__':

    # Set the geometric parameters
    width = 300
    height = 300

    # Set the division parameters of the domain for parallelization
    N = 2 # set the number of subdomains in the x direction 
    M = 2 # set the number of subdomains in the y direction 
    
    # set kinematic viscosity
    v = 0.03

    # set the velocity wall
    u_wall = np.array([0.1, 0]) # Change the first argument to determine the velocity of the lid in the x direction 

    # Set the time
    time_steps = 25000
    plot_time_steps = 500

    # Set the output folder to save the pictures there
    output_loc = os.getcwd() + "/Milestone7/" 

    # Print out the Reynolds number 
    Re = calculate_reynolds_number(width,height,u_wall[0],v)
    #print(f"Reynolds number Re: {Re}")

    # Initiate the parallel solution
    parallelization(width,height,N,M,v,u_wall,time_steps,plot_time_steps)
