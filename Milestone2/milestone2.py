#######
## import general modules
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
######

#######
## call the functions from other scripts
sys.path.append('../HPC_Submission/Milestone1')  # to call the functions from milestone1
import milestone1
######

# Milestone 2: Collision Operator

def get_weights(): 
    """
    This function returns the weights for the prob. dens. func. for the simulations. 
    Returns: 
        wi: np.ndarray 
            The weights stated on the Milestone 2 that has a size of 1 x 9
    """
    wi =  np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
    return wi

def calculate_f_equilibrium(rho:np.ndarray, u:np.ndarray): 
    """
    This function returns prob. density funct. for equilibrium feq that has a size of 9 x width x height. 
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
    c = milestone1.get_velocity_set()

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
    This function returns prob. density funct. after collision using Milestone 2 equation 2. \n
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
    rho = milestone1.calculate_density(f)
    # Calculate the velocity field before collision
    u = milestone1.calculate_velocity_field(f)
    # Calculate the equilibrium prob. density funct. 
    feq = calculate_f_equilibrium(rho,u)
    # Obtain the pdf. after collision
    f_collision = f + w * (feq-f) 
    return f_collision

def time_iterative(f_init:np.ndarray,w:float,time_steps:int,output_loc:str=""): 
    """
    This function iterates over time steps by performing streaming and collision for one time step.\n
    Args: 
        f_init: np.ndarray
            The initial probability density function that has a size of 9 x width x height
        w: float
            Relaxation parameter between 0 and 2.
        time_steps: int
            Time steps taken during the simulation. One time step is one streaming and one collision.
        output_loc: str
            The location of the output file.
    Returns: 
        f_final: np.ndarray
            The final prob. dist. funct. that has a size of 9 x width x height
    """
    # Call the velocity set
    c = milestone1.get_velocity_set()
    # Call the weights for D2Q9 lattice
    wi = get_weights()

    # Get the width and height from the pdf
    width = f_init.shape[1]
    height = f_init.shape[2]

    # Running the simulations for time_steps time
    for time_step in range(0,time_steps):
        
        # Calculate density 
        rho_init = milestone1.calculate_density(f_init)
        # Visualize density 
        title = f"Density at t={time_step}"
        output_name = f"density_width_{width}_height_{height}_time_{time_step}"
        milestone1.visualize_density(rho_init,title=title,output_name=output_name,output_loc=output_loc)

        # Streaming step
        f_streaming = milestone1.streaming(f_init)

        # Collision step
        f_collision = collision(f_streaming,w)
       
        # Check the mass conservation 
        milestone1.check_mass_conservation(f_init,f_collision,f"For the time step:{time_step} (streaming + collision)")

        f_init = f_collision


    f_final = f_init
      
    return f_final


if __name__ == '__main__':

    # Define the domain sizes
    width = 15
    height = 10
    # Set the relaxation parameter that is needed for collision
    w = 1
    # Set the time steps for the simulation
    time_steps = 50
    # Set the path to save the pictures
    path = os.getcwd() + "/Milestone2"

    test_number = int(input("Choose which test that you want to run: \n" + 
    "1. Create a uniform density on your grid and set the density to a slightly higher value at the center. What happens? \n" + 
    "2. Choose an initial distribution of rho(r) and u(r) at t = 0. What happens dynamically as well as in the long time limit t --> inf. \n " +
    "Enter your input(either 1 or 2): "))

    if test_number == 1: 

        # Set the path to save the pictures
        path = path + "/Test1/"

        # Create and modify the prob. dist. function, if needed...
        # initialize f: the probability density function
        f_initial = milestone1.initialize_pdf_zeros(width,height)
        # modify f_initial
        # Putting one particle to 1st channel of all nodes
        f_initial[1,:,:] = 10
        # Putting one particle to 2nd channel of all nodes
        f_initial[2,:,:] = 10
        # Putting 50 particles to 1st channel of the node at 7,4
        f_initial[1,7,4] = 20
        
        # running the simulation time_steps time
        time_iterative(f_initial,w,time_steps,output_loc=path)
    
    if test_number == 2: 

        # Set the path to save the pictures
        path = path + "/Test2/"

        # Define the parameters for the Gaussian distribution for initial_rho
        mean = 0.0
        std_dev = 1.0
        size = (width, height)  # Desired shape of the 2D array

        # Create the 2D array of rho_initial with Gaussian distribution
        rho_initial = np.random.normal(mean, std_dev, size)

        # Clip the values to be within the range of 0 to 0.99 for rho_initial
        rho_initial = np.clip(rho_initial, 0, 0.99)

        # Define the parameters for the uniform distribution for initial_u
        low = 0.0  # Lower bound of the range
        high = 0.1  # Upper bound of the range
        size = (2,width, height)  # Desired shape of the 2D array

        # Create the 2D array  of u_initial with uniform distribution 
        u_initial = np.random.uniform(low, high, size)

        # Start with collision step to create initial pdf
        # collision step 
        f = calculate_f_equilibrium(rho_initial,u_initial)
       
        # running the simulation time_steps time
        time_iterative(f,w,time_steps,output_loc=path)




