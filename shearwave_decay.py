import sys
import os
import numpy as np
############### CHANGE THE PARAMETERS BELOW (IF NEEDED) AND RUN THE SCRIPT ####################
"""NOTING THAT: THE USER IS ASKED WHICH TEST TO RUN, SIMPLY ENTER 1 OR 2. (after running the script)
        1 -> FOR THE SINUSOIDAL DENSITY DISTURPTION
        2 -> FOR THE SINUSOIDAL VELOCITY DISTURPTION"""
# Geometric entities of the domain
width = 100
height = 50
epsilon = 0.05 # scaling factor of the sinusiodals functions (for both cases - )
rho_0 = 0.075 # initial density (for case 1)
# Define time steps 
time_steps = 3500
plot_time_steps = 250
# Relaxation parameters to be used
w_vals = np.arange(0.2,2,0.4)
################################################################################################
################################################################################################
################################################################################################
#################### BELOW ALL THE FUNCTIONS THAT ARE NEEDED FOR THIS SCRIPT ###################
######### import general modules
import matplotlib.pyplot as plt
import math
def shearwave_decay(width:int,height:int,w_vals:np.ndarray,time_steps:int,plot_time_steps:int,output_loc:str=None,epsilon:float=0.05,rho_0:float=0.075): 
    """
    This function simulates shear wave decay phenomenon, and returns the final probability density function. 
    NOTING THAT: THE USER IS ASKED WHICH TEST TO RUN, SIMPLY ENTER 1 OR 2. 
        1 -> FOR THE SINUSOIDAL DENSITY DISTURPTION
        2 -> FOR THE SINUSOIDAL VELOCITY DISTURPTION
    Also, it plots&saves
        a. the maximum velocity or density amplitudes over time. 
        b. the sinusoidal velocity profiles over time (if 2 is chosen).
        c. comparison of analytical and measured viscosity. (if 2 is chosen).
    Args: 
        width: int 
            The length of the domain in the x direction. 
        height: int 
            The length of the domain in the y direction. 
        w_vals: np.ndarray 
            Relaxation parameters to be used in the simulation as an array list. 
        time_steps: int
            The total time to run the simulation. 
        plot_time_steps: int
            In each plot_time_steps, the streamplot is created & saved. 
        output_loc: str
            The pictures are saved at os.getcwd() + output_loc. If it is not given, pictures are saved in the current directory in GeneratedResults folder that is created.
        epsilon: float 
            Scaling factor of the sinusiodals functions, if not given default value is 0.05.
        rho_0: float 
            The initial density value,if not given default value is 0.075.
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

    test_number = int(input("Choose which test that you want to run: \n" + 
                            "1. rho(r,0) = rho_0 + epsilon * sin(2*pi*x/L_x) and u(r,0) = 0. Observe what happens with 2D density distribution in time.  \n" + 
                            "2. rho(r,0) = 1 and u_x(r,0) = epsilon * sin(2*pi*y/L_y). Observe what happens in the long time limit. \n" +
                            "Assume Stokes flow. Calculate kinematic viscosity. How does kinematic viscosity scale with w? \n" + 
                            "Enter your  your input (either 1 or 2): "))

    # Create w_v_measured_dict that measures the kinematic viscosity through the simulation
    w_v_measured_dict = dict()
    # Call the velocity set
    c = get_velocity_set()
    # Call the weights for D2Q9 lattice
    wi = get_weights()
    
    if test_number == 1:
        ###################################################################
        # Roadmap Nummer 1 - SINUSOIDAL DENSITY / ZERO VELOCITY
        # Choose an initial distribution such that 
        # rho(r,0) = rho_0 + epsilon * sin(2*pi*x/L_x) and 
        # u(r,0) = 0
        ############## Iterating for different w values ##############
        for w in w_vals:
            ############## Create initial arrays ##############
            # Create velocity field u(r,0) = 0
            u_initial = np.zeros((2, width, height))          
            # Create density rho(r,0) 
            rho_initial = np.ones((width, height))
            # Generate x-axis indices
            x_coordinates = np.arange(rho_initial.shape[0])[:, np.newaxis]
            # Multiply rho_initial by x-axis indices using broadcasting 
            rho_initial = rho_0 + epsilon * np.sin(rho_initial * x_coordinates * 2 * math.pi / (width-1))
            # Calculating the initial prob. density func.
            f = calculate_f_equilibrium(rho_initial,u_initial)

            # Create f_time to store the pdfs over time
            f_time = np.zeros((time_steps+1,9,width,height))
            f_time[0,:,:,:] = f

            # running the simulation time_steps time
            for time_step in range(time_steps):
                if time_step % plot_time_steps == 0:
                    print("Time step:",time_step)
                # Streaming step
                f_streaming = streaming(f)
                # Collision step
                f_collision = collision(f_streaming,w)
                # Equate f_collision to f to iterate again
                f = f_collision
                # Save the pdf in f_time
                f_time[time_step+1,:,:,:] = f

            # Visualize the maximum density amplitudes change wrt time
            # Also get the maximum amplitudes and time steps as an output
            output_name = f"maximum_density_amplitudes_width_{width:.2f}_height_{height:.2f}_w_{w:.2f}"
            max_amplitudes, t_coordinates = visualize_max_sinusoidal_density_amplitude_over_time(f_time,w,output_name=output_name,output_loc=output_loc)

    if test_number == 2: 
        ###################################################################
        # Roadmap Nummer 2 - SINUSOIDAL VELOCITY
        # Choose an initial distribution such that  
        # rho(r,0) = 1 and
        # u_x(r,0) = epsilon * sin(2*pi*y/L_y) 
        ############## Iterating for different w values ##############
        for w in w_vals:
            ############## Create initial arrays ##############
            # Create rho(r,0) = 1
            rho_initial = np.ones((width,height))
            # create u_x(r,0) = epsilon * sin(2*pi*y/L_y)
            u_initial = np.ones((2,width,height))
            u_initial[1,:,:] = 0
            # Get the number of columns
            num_columns = u_initial[0,:,:].shape[1]
            # Generate the desired matrix using broadcasting
            vect = np.arange(num_columns)
            # y coordinate dependent matrix 
            res = np.tile(vect, (u_initial[0,:,:].shape[0], 1))
            # Multiply u_x by y-axis indices using broadcasting
            u_initial[0,:,:] = epsilon * np.sin(res * 2 * math.pi / height)     
            # Calculating the initial prob. density func.
            f = calculate_f_equilibrium(rho_initial,u_initial)
            # Create f_time to store the pdfs over time
            f_time = np.zeros((time_steps+1,9,width,height))
            f_time[0,:,:,:] = f 

            # running the simulation time_steps time
            for time_step in range(time_steps):
                if time_step % plot_time_steps == 0:
                    print("Time step:",time_step)
                # Streaming step
                f_streaming = streaming(f)
                # Collision step
                f_collision = collision(f_streaming,w)
                # Equate f_collision to f to iterate again
                f = f_collision
                # Save the pdf in f_time
                f_time[time_step+1,:,:,:] = f

            # Visualize the sinusoidal velocity the profile in the x direction
            output_name = f"velocity_profile_sinusoidal_width_{width:.2f}_height_{height:.2f}_w_{w:.2f}"
            visualize_sinusoidal_velocity_amplitude_over_time(f_time,w,plot_time_steps,output_name=output_name,output_loc=output_loc)

            # Visualize the maximum velocity amplitudes change in the x direction wrt time
            # Also get the maximum amplitudes and time steps as an output
            output_name = f"maximum_velocity_amplitudes_width_{width:.2f}_height_{height:.2f}_w_{w:.2f}"
            max_amplitudes, t_coordinates = visualize_max_sinusoidal_velocity_amplitude_over_time(f_time,w,output_name=output_name,output_loc=output_loc)

            # Calculate Measured Viscosity
            v_measured = calculate_measured_viscosity(max_amplitudes,height)
            w_v_measured_dict[w] = v_measured

        # Plot measured kinematic viscosity vs analytical solution
        plot_measured_vs_analytical_viscosity(w_v_measured_dict,output_name="analytical_vs_measured_viscosity",output_loc=output_loc)

        return f_time[-1,:,:,:] # final pdf 
###
def visualize_sinusoidal_density_amplitude_over_time(f_time:np.ndarray,w:float,plot_time_steps:int,title:str="Density Amplitudes vs x coordinates",output_name:str="output",output_loc:str=None): 
    """
    This function plots time-Wise sinusoidal amplitude of density.  \n
    Args: 
        f_time: np.ndarray
            The prob. density functions over the time. It has a shape of  time x 9 x width x height
        w: float
            Relaxation parameter for the plot
        plot_time_steps:int
            The plot is given in each plot_time_steps.
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        output_loc: str
            The location of the output file.
    Returns: 
        None
    """
    # Set the figure size
    plt.figure(figsize=(16, 9))  

    # Obtain the time steps & height & width & center in the y direction
    time_steps = f_time.shape[0]
    height = f_time.shape[3]
    width = f_time.shape[2]
    y_center = int(height/2)

    # Create t-coordinates array for the time steps
    t_coordinates = np.arange(width)

    # Calculate density at y=height/2
    for time in range(0, time_steps):
        if ((time+1) % plot_time_steps == 0) or (time == 0):
            # Calculate the density for the time step time
            rho = milestone1.calculate_density(f_time[time,:,:,:])
            # Get the density values at y_center
            density_values = abs(rho[:,y_center])
            # Set the labels
            if time == 0:
                label_str =  f"Time:{time}"
            else: 
                label_str =  f"Time:{time+1}"
            # Plot
            plt.plot(t_coordinates, density_values, label=label_str)

    # Set labels & title & legend
    plt.xlabel('x coordinates')
    plt.ylabel('Density amplitudes')
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='lower left')
    # Put the annotation showing relaxation parameter
    plt.text(0.95, 0.95, f"Relaxation parameter w={w:.2f}", ha='right', va='top',fontweight="bold", transform=plt.gca().transAxes)
    plt.text(0.95, 0.9, f"Width={width:.2f}, Height:{height:.2f}", ha='right', va='top',fontweight="bold", transform=plt.gca().transAxes)

    # Save the image 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone3/Test1/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name)
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name = output_loc + output_name + ".png"
        plt.savefig(file_name)
        print(f"Picture saved at location: {file_name}")

    # Show the plot
    #plt.show()

def visualize_max_sinusoidal_density_amplitude_over_time(f_time:np.ndarray,w:float,title:str=None,output_name:str="output",output_loc:str=None): 
    """
    This function plots maximum sinusoidal amplitude of density over time. \n
    Args: 
        f_time: np.ndarray
            The prob. density functions over the time. It has a shape of  time x 9 x width x height
        w: float
            Relaxation parameter to be shown as an annotation
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        output_loc: str
            The location of the output file.
    Returns: 
        max_amplitudes: np.ndarray
            Maximum amplitudes of the density over the time steps.
        x_coordinates: np.ndarray
            time steps of the simulation.
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
    plt.rc('legend', fontsize=28)  

    # Obtain the time steps & width & center in the x direction
    time_steps = f_time.shape[0]
    height = f_time.shape[3]
    y_center = int(height/2)

    # Create t-coordinates array for the time 
    t_coordinates = np.arange(time_steps)
    
    # Create a list to contain max amplitude values over time
    max_amplitudes = []

    # Calculate y-coordinates
    for time in range(0, time_steps):
        # Calculate the density for the time step time
        rho = calculate_density(f_time[time,:,:,:])
        # Get the density values at y_center
        density_values = abs(rho[:,y_center])
        max_amplitude = np.max(density_values)
        max_amplitudes.append(max_amplitude)

    max_amplitudes = np.array(max_amplitudes)

    # Plot the line graph for amplitude vs time 
    plt.plot(t_coordinates, max_amplitudes, linewidth=3)

    # Show all the plots
    plt.xlabel('Time (s)')
    plt.ylabel('Maximum density amplitude')
    if title != None:
        plt.title(title)
    plt.grid(True)

    # Put the annotation showing relaxation parameter / width / height
    plt.text(0.95, 0.95, f"Relaxation parameter ω:{w:.2f}", ha='right', va='top',fontweight="bold", fontsize=24, transform=plt.gca().transAxes)
    plt.text(0.95, 0.9, f"Width:{width:.2f}, Height:{height:.2f}", ha='right', va='top',fontweight="bold", fontsize=24, transform=plt.gca().transAxes)

    # Save the image 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone3/Test1/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name =  output_loc + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")

    # Show the figure
    #plt.show()

    return max_amplitudes, t_coordinates

def visualize_sinusoidal_velocity_amplitude_over_time(f_time:np.ndarray,w:float,plot_time_steps:int,title:str=None,output_name:str="output",output_loc:str=None): 
    """
    This function plots time-Wise sinusoidal amplitude of velocity in the x direction.  \n
    Args: 
        f_time: np.ndarray
            The prob. density functions over the time. It has a shape of  time x 9 x width x height
        w: float
            Relaxation parameter for the plot
        plot_time_steps:int
            The plot is given in each plot_time_steps.
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        output_loc: str
            The location of the output file.
    Returns: 
        None
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

    # Obtain the time steps & height & width & center in the x direction
    time_steps = f_time.shape[0]
    height = f_time.shape[3]
    width = f_time.shape[2]
    x_center = int(width/2)

    # Create y-coordinates array
    y_coordinates = np.arange(height)

    # Calculate velocity amplitudes
    for time in range(0, time_steps):
        if ((time+1) % plot_time_steps == 0) or (time == 0):
            # Calculate the velocity field u
            u = calculate_velocity_field(f_time[time,:,:,:])
            # Get the velocity field in the x direction wrt y coordinates at x_center
            velocity_amplitudes = u[0,x_center,:]
            # Plot the sinusoidal line graph
            if time == 0:
                label_str =  f"Time:{time}"
            else: 
                label_str =  f"Time:{time+1}"
            plt.plot(velocity_amplitudes, y_coordinates, label=label_str)

    # Set Labeling & Title & Grid & Legends 
    plt.xlabel('Velocity in x direction')
    plt.ylabel('y coordinates')
    if title!= None:
        plt.title(title)
    plt.grid(True)
    plt.legend(loc='lower left')
    # Put the annotation showing relaxation parameter / width / height
    plt.text(0.95, 0.95, f"Relaxation parameter ω:{w:.2f}", ha='right', va='top',fontweight="bold", fontsize=24, transform=plt.gca().transAxes)
    plt.text(0.95, 0.9, f"Width:{width:.2f}, Height:{height:.2f}", ha='right', va='top',fontweight="bold", fontsize=24, transform=plt.gca().transAxes)

    # Save the figure 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone3/Test2/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name =  output_loc + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")

    # Show the figure
    #plt.show()

def visualize_max_sinusoidal_velocity_amplitude_over_time(f_time:np.ndarray,w:float,title:str=None,output_name:str="output",output_loc:str=None): 
    """
    This function plots maximum velocity amplitude of density over time. \n
    Args: 
        f_time: np.ndarray
            The prob. density functions over the time. It has a shape of  time x 9 x width x height
        w: float
            Relaxation parameter for the plot
        title: str
            Title of the plot
        output_name: str
            The name of the file to be saved. 
        output_loc: str
            The location of the output file.
    Returns: 
        max_amplitudes: np.ndarray
            Maximum amplitudes of the velocity over the time steps.
        t_coordinates: np.ndarray
            time steps of the simulation.
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
    plt.rc('legend', fontsize=28)  

    # Obtain the time steps & width & center in the x direction
    time_steps = f_time.shape[0]
    width = f_time.shape[2]
    x_center = int(width/2)

    # Create t-coordinates array for the time 
    t_coordinates = np.arange(time_steps)
    
    # Create a list to contain max velocity amplitude values over time
    max_amplitudes = []

    # Calculate x-coordinates
    for time in range(0,time_steps):
        # Calculate the velocity field for each time step
        u = calculate_velocity_field(f_time[time,:,:,:])
        # Get the velocity field in the x direction wrt y coordinates at x_center
        velocity_amplitudes = u[0,x_center,:]
        # Taking the absolute value of the velocity field in the x direction wrt y coordinates at x_center
        velocity_amplitudes = np.abs(velocity_amplitudes)
        # Get the maximum velocity value during the time step time
        max_amplitude = np.max(velocity_amplitudes)
        max_amplitudes.append(max_amplitude)
    max_amplitudes = np.array(max_amplitudes)

    # Plot the line graph for amplitude vs time 
    plt.plot(t_coordinates, max_amplitudes,linewidth=3)

    # Set Labeling & Title & Grid  
    plt.xlabel('Time (s)')
    plt.ylabel('Maximum velocity amplitude')
    plt.grid(True)
    if title!= None:
        plt.title(title)
    # Put the annotation showing relaxation parameter / width / height
    plt.text(0.95, 0.95, f"Relaxation parameter ω:{w:.2f}", ha='right', va='top',fontweight="bold", fontsize=24, transform=plt.gca().transAxes)
    plt.text(0.95, 0.9, f"Width:{width:.2f}, Height:{height:.2f}", ha='right', va='top',fontweight="bold", fontsize=24, transform=plt.gca().transAxes)

    # Save the figure 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone3/Test2/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name =  output_loc + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")

    # Show the figure
    #plt.show()

    return max_amplitudes, t_coordinates

def calculate_analytical_viscosity(w_array:np.ndarray):
    """
    This function creates&returns a dictionary that maps w values to v values (analytical).\n
    Args:
        w_list: np.ndarray 
            The array containing the w values (relaxation parameter) for the simulation. 
    Returns: 
        w_v_analytical_dict: dictionary
            The dictionary that has the relaxation parameter as keys, and kinematic viscosity as values.
    """
    w_v_analytical_dict = dict()
    for w in w_array: 
        v = 1/3 * (1/w - 1/2)
        w_v_analytical_dict[w] = v
    return w_v_analytical_dict

def calculate_measured_viscosity(amplitude_array:np.ndarray,length:float):
    """
    This function returns the kinematic viscosity for given relaxation parameter using the formula v = (ln(a0) - ln(a(t1))) / ((2*pi/L)^2 * t1) \n
    Args:
        amplitude_array: np.ndarray
            that contains the values of amplitudes for the simulation
        length: float
            that is the height of a channel (L in the formula)
    Returns: 
        v: float
            Measured kinematic viscosity
    """
    a0 = amplitude_array[0]
    a_t1 = amplitude_array[int(len(amplitude_array)*4/5)]
    t1 =  len(amplitude_array)
    v = (math.log(a0) - math.log(a_t1)) / (t1 * (2*math.pi/length)**2)
    return v

def plot_measured_vs_analytical_viscosity(w_v_measured_dict:dict,output_name:str="output",output_loc:str=None,title:str=None): 
    """
    This function plots relaxation parameter (w) vs kinematic viscosity (v) for both measured values and analytical solution. \n
    Args:
        measured_viscosity: dictionary
            The dictionary mapping w values to v values that is obtained through the simulations.
        output_name: str
            The name of the file to be saved. 
        output_loc: str
            The location of the output file.
    Returns: 
        None
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
    plt.rc('legend', fontsize=28)    

    # Plotting w vs v values (from the simulations) 
    # Separate the keys and values from the dictionary
    keys_measured = list(w_v_measured_dict.keys()) 
    # same keys to be used to calculate analytical viscosity 
    w_values = keys_measured
    values_measured = list(w_v_measured_dict.values())
    # Plotting the line chart for the measured kinematic viscosity vs relaxation parameter
    plt.plot(keys_measured, values_measured, marker='o',color="red",label="Measured",linewidth=3)

    # Obtain the analytical solution 
    w_v_analytical_dict = calculate_analytical_viscosity(w_values)
    # Separate the keys and values from the dictionary
    keys_analytical = list(w_v_analytical_dict.keys())
    values_analytical = list(w_v_analytical_dict.values())
    # Plotting the line plot for the analytical case
    plt.plot(keys_analytical, values_analytical, marker='o',color="blue",label="Analytical",linewidth=3)

    # Set Labeling & Title & Grid  
    plt.xlabel('Relaxation parameter ω')
    plt.ylabel('Kinematic viscosity v')
    if title!= None:
        plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)

    # Save the figure 
    if output_loc == None: 
        dir = os.getcwd()
        file_path = dir + "/Milestone3/Test2/"
        file_name = file_path + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")
    else: 
        file_name = output_loc + output_name + ".png"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"Picture saved at location: {file_name}")

    #plt.show()

def get_velocity_set():
    """
    This function returns the velocity set for the simulations. \n
    Returns: 
        c: np.ndarray 
            The velocity set stated on the Milestone 1 that has a size of 2 x 9
    """
    # Create the velocity set c
    c  = np.transpose(np.array([[0,1,0,-1,0,1,-1,-1,1],[0,0,1,0,-1,1,1,-1,-1]]))
    return c 

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
    rho = calculate_density(f)
    # Calculate the velocity field before collision
    u = calculate_velocity_field(f)
    # Calculate the equilibrium prob. density funct. 
    feq = calculate_f_equilibrium(rho,u)
    # Obtain the pdf. after collision
    f_collision = f + w * (feq-f) 
    return f_collision

def streaming(f:np.ndarray):
    """
    This function calculates & returns the probability density f after streaming step using np.roll. \n
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
    This functions sums up all of the particles in the 9 channels for a given prob. density funct. f along axis 0, and returns the density. \n
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
    This function calculates & returns the velocity field for a given prob. density funct. f.\n
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


if __name__ == '__main__':
    shearwave_decay(width,height,w_vals,time_steps,plot_time_steps,epsilon=epsilon,rho_0=rho_0)
   
    


           
            
            

        