# Milestone 1: Introduction to LBE and Streaming Operator
import numpy as np
import matplotlib.pyplot as plt
import os

### TO DO: SHOW WALLS AT LEFT AND RIGHT BOUNDARIES
### TO DO: X coord , ycoord

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
 
def initialize_pdf_zeros(width:int,height:int):
    """
    This function initializes&returns the probability density function f whose elements are all zeros. \n
    Args:
        width: int
            The size of the domain in the x direction
        height: int 
            The size of the domain in the y direction
    Returns: 
        f: np.ndarray
            The probability density function that has a size of 9 x width x height
    """
    # Create the probability density function f that is all zeros
    f = np.zeros((9,width,height)) 
    return f

def initialize_rho_zeros(width:int,height:int):
    """
    This function initializes&returns the density rho whose elements are all zeros. \n
    Args:
        width: int
            The size of the domain in the x direction
        height: int 
            The size of the domain in the y direction
    Returns: 
        rho: np.ndarray
            The density that has a size of width x height
    """
    # Create the density rho that is all zeros
    rho = np.zeros((width,height)) 
    return rho

def initialize_u_zeros(width:int,height:int):
    """
    This function initializes&returns the velocity field v whose elements are all zeros. \n
    Args:
        width: int
            The size of the domain in the x direction
        height: int 
            The size of the domain in the y direction
    Returns: 
        u: np.ndarray
            The velocity field that has a size of 2 x width x height. 
            If first axis 0, it is the velocity in the x direction. 
            If first axis 1, it is the velocity in the y direction. 
    """
    # Create the velocity field u that is all zeros
    u = np.zeros((2,width,height)) 
    return u

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

def check_mass_conservation(old_pdf:np.ndarray,new_pdf:np.ndarray,event:str="",difference:float=2): 
    """
    This function calculates the sum of two different pdfs & prints out if the difference between them is bigger than difference. \n
    Args: 
        old_pdf: np.ndarray
            The old probability density function that has a size of 9 x width x height
        new_pdf: np.ndarray
            The new probability density function that has a size of 9 x width x height
        event: str
            The name of the simulation. For example, Couette, Poiseuille etc...
        difference:float
            The threshold of the difference as a percentage.
    Returns: 
        None.
    """
    if (abs(new_pdf.sum() -  old_pdf.sum())) / (abs(new_pdf.sum())) < difference:
        print(f"Mass is conserved for the event {event}")
    elif (abs(new_pdf.sum() - old_pdf.sum())) / (abs(new_pdf.sum()))  > difference:
        print(f"Mass is not conserved for the event {event}")
        print("Sum of the old prob. density func.: ",old_pdf.sum())
        print("Sum of the new prob. density func.: ",new_pdf.sum())
        print()

def visualize_density(rho:np.ndarray,title:str="",output_name:str="output",show_colorbar:bool=True,add_wall:bool=False,pos:list=None,output_loc:str=None):
    """
    This function plots the density over the domain and saves it as .png in the same directory where the code is run. \n
    Args: 
        rho: np.ndarray
            The density that has a size of width x height
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
    heatmap = ax.imshow(rho.transpose())

    # Add colorbar
    if show_colorbar == True:
        cbar = plt.colorbar(heatmap)

    # Invert the y-axis
    ax.invert_yaxis()

    # Set the title
    ax.set_title(title)

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
        file_name = output_loc + output_name +".png"
        plt.savefig(file_name)
        print(f"Picture saved at location: {file_name}")
   
    # Show the plot
    #plt.show()

    return fig, ax

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

if __name__ == '__main__':
   
    # Define the domain sizes
    width = 15
    height = 10

    # Create and modify the prob. dist. function, if needed...
    # initialize f: the probability density function
    f_initial = initialize_pdf_zeros(width,height)
    # modify f_initial
    # Putting one particle to 1st channel of all nodes
    f_initial[1,:,:] = 1
    # Putting one particle to 2nd channel of all nodes
    f_initial[2,:,:] = 1
    # Putting 50 particles to 1st channel of the node at 7,4
    f_initial[1,7,4] = 500
    
    # Calculate the initial density
    rho = calculate_density(f_initial)
    # Visualize the density
    title = f"Initial density (width= {width}, height={height})"
    output_name = f"Initial_density_width_{width}_height_{height}"
    visualize_density(rho, title=title,output_name=output_name)
    
    # Calculate the velocity field
    velocity_field = calculate_velocity_field(f_initial)
    # Visualize the velocity field in the x direction
    title = f"Initial velocity field in x"
    output_name = f"Initial_velocity_field_in_x_width_{width}_height_{height}"
    visualize_velocity_field_x(velocity_field, title=title,output_name=output_name)
    # Visualize the velocity field by streamplot
    title = f"Initial velocity field"
    output_name = f"Initial_velocity_field_stream_plot"
    streamplot_velocity(velocity_field, title=title,output_name=output_name)
 
    # Streaming step
    f_stream = streaming(f_initial)
    # Check the mass conservation after streaming
    check_mass_conservation(f_initial,f_stream,"Streaming")
    # Calculate the density after streaming
    rho_stream = calculate_density(f_stream)
    # Calculate the density after streaming
    title = f"After streaming density (width= {width}, height={height})"
    output_name = f"After_streaming_density_width_{width}_height_{height}"
    visualize_density(rho_stream,title=title,output_name=output_name) 
    # Calculate the velocity field
    velocity_field = calculate_velocity_field(f_stream)
    # Visualize the velocity field in the x direction
    title = f"After streaming velocity field in x"
    output_name = f"After_streaming_velocity_field_in_x_width_{width}_height_{height}"
    visualize_velocity_field_x(velocity_field, title=title,output_name=output_name)
    
    # Visualize the velocity field by streamplot
    title = f"After streaming velocity field"
    output_name = f"After_streaming_velocity_field_stream_plot"
    streamplot_velocity(velocity_field, title=title,output_name=output_name)

    """ #### Velocities in the y direction ###
    # Commented out bcs initial pdf created such that channel 1 (x direction) at x = 7, y = 4 dominates the streaming 
    # Visualize the velocity field in the y direction before streaming
    title = f"Initial velocity field in y"
    output_name = f"Initial_velocity_field_in_y_width_{width}_height_{height}"
    visualize_velocity_field_y(velocity_field, title=title,output_name=output_name)
    # Visualize the velocity field in the y direction after streaming
    title = f"After streaming velocity field in y"
    output_name = f"After_streaming_velocity_field_in_y_width_{width}_height_{height}"
    visualize_velocity_field_y(velocity_field, title=title,output_name=output_name) """
