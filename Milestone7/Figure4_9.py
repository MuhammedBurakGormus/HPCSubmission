import matplotlib.pyplot as plt

measured_times = {
    1 : "49:30", 
    2 : "09:35", 
    3 : "04:55",
    4 : "03:39",
    5 : "03:28",
    6 : "03:28",
    7 : "02:57",
    8 : "02:52",
    9 : "02:58",
    10 : "02:51"
}

# Function to convert to seconds
def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(":"))
    total_seconds = minutes * 60 + seconds
    return total_seconds

# Create a new dictionary with time values in seconds
measured_times_seconds = {key: time_to_seconds(value) for key, value in measured_times.items()}

# Set the number of grid points (we use 300x300 domain)
no_grid_points = 300*300

# Set the time steps (we run the code 100000 unit times)
time_steps = 100000

# Create final dictionary containing MLUPs
final_dict = dict()
for CPU in measured_times_seconds: 
    run_time = measured_times_seconds[CPU]
    MLUPs = no_grid_points * time_steps / run_time
    final_dict[CPU] = MLUPs


# Extract keys and values
keys = list(final_dict.keys())
values = list(final_dict.values())


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

plt.plot(keys, values, linestyle='-', color='b',linewidth=5)
plt.scatter(keys,values,color='b',s=150)
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.xlabel('# of CPUs')
plt.ylabel('MLUPS')
plt.xticks(keys, keys)  # Set x-axis ticks to match keys
plt.annotate('Log-Log Scale', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=22,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"))
#plt.show()
plt.savefig('scaling.png')






  