import numpy as np
import matplotlib.pyplot as plt
import cv2



def create_grid(obst_idx,valid_file,invalid_file):
    x_invalid = []
    y_invalid = []

    for line in invalid_file:
        
        words = line.split(',')  
        x_invalid.append(int(float(words[0])))
        y_invalid.append(int(float(words[1])))

    x_valid= []
    y_valid = []

    for line in valid_file:
        words = line.split(',')  
        x_valid.append(int(float(words[0])))
        y_valid.append(int(float(words[1])))

    x_valid = np.array(x_valid)
    x_invalid = np.array(x_invalid)
    y_valid = 399- np.array(y_valid)
    y_invalid = 399- np.array(y_invalid)


    grid[y_valid,x_valid] = 0
    grid[y_invalid,x_invalid] = 0.5
    grid[obst_idx[:,0],obst_idx[:,1]] = 1

    temp = np.empty([grid.shape[0],grid.shape[1],3])

    temp[y_valid,x_valid,:] = [39, 174, 96]
    temp[y_invalid,x_invalid,:] = [219, 225, 229]
    temp[obst_idx[:,0],obst_idx[:,1],:] = [0,0,0]

    return temp







env_file = open("../custom_maps/env4.txt")

invalid_file = open("../stats/invalid_states.txt",'r')
valid_file = open("../stats/valid_states.txt",'r')

predicted_invalid_file = open("../stats/predicted_invalid_states.txt",'r')
predicted_valid_file = open("../stats/predicted_valid_states.txt",'r')
env_list = []
grid = np.zeros([400,400])
grid_predicted = np.zeros([400,400])

#loading the environment file
val = 0
for line in env_file:
    val += 1
    if(val ==1):
        continue
    temp_list =[]
    
    for word in line:
        if word == '\n':
            break;
        temp_list.append(int((word)))
   
    env_list.append(temp_list)
        

env = np.array(env_list)
obst_idx = np.argwhere(env != 0)
fig,ax = plt.subplots(2)
temp = create_grid(obst_idx,valid_file,invalid_file)
temp_predicted = create_grid(obst_idx,predicted_valid_file,predicted_invalid_file)
ax[0].imshow(temp/255,alpha=1.0)
ax[1].imshow(temp_predicted/255)
# ax.axis('off')
plt.show()






