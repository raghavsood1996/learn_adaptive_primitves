import numpy as np 
import matplotlib.pyplot as plt



data = np.load("../model_data/data_float360_8_fg_cs.npy")

data_shape = np.shape(data)
dist_list = []
inf_list = []
for i in range(data_shape[1]):
    dist = int(data[-2,i])
    inf = data[-1,i]
    dist_list.append(dist)
    inf_list.append(inf)


disc_r_count = np.zeros([np.max(dist_list)+1])
disc_pos_count = np.zeros([np.max(dist_list)+1])

for i in range(len(dist_list)):
    
    disc_r_count[dist_list[i]] +=1 
    if(inf_list[i] == 1):
        disc_pos_count[dist_list[i]] +=1 
        


plt.plot(np.linspace(0,np.max(dist_list)+1,np.max(dist_list)+1),disc_r_count,disc_pos_count)
plt.show()
