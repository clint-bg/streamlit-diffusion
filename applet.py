import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Example Diffusion Simulator')
st.markdown('This is a simple example of a diffusion simulator. The user can \
select the number of random walks in the simulation using the slider below.')

val = st.slider('Select a value for the number of random walks', 20, 200, 100)

# Create the simulation with the provided slider value
nsteps = 100; nwalks = val
sqdists = np.zeros((nwalks,nsteps))
for i in range(nwalks):
    #set parameters for 2-D random walk
    x = 0; y = 0; z = 0 #initial x,y, and z coordinates
    x_loc = [x]; y_loc = [y]; z_loc = [z]
    for j in range(nsteps):
        # for each step, the x or y coordinates change by 1 or -1 depending on the random number
        direction = np.random.choice([-1,0,1])
        if direction == -1:
            x += np.random.choice([-1,1])
        elif direction == 1:
            y += np.random.choice([-1,1])
        else:
            z += np.random.choice([-1,1])
        sqdists[i,j] = (x**2 + y**2 + z**2) #distance from origin
        x_loc.append(x); y_loc.append(y); z_loc.append(z)


#plot one of the simulation runs (the last one)
fig, ax = plt.subplots()
#format the text to a smaller size
ax.rc('font', size=8)
ax.plot(x_loc,z_loc)
ax.plot(x_loc[0],z_loc[0],'go')
ax.plot(x_loc[-1],z_loc[-1],'ro')
ax.set_xlabel('x position'); ax.set_ylabel('z position')
ax.set_title('3-D Random Walk plotted in 2D')
ax.legend(['Random Walk','Start','End'])
st.pyplot(fig)


#calculate the average squared distance from the origin
avg_sqdist = np.mean(sqdists, axis=0)
df = pd.DataFrame({'Number of steps': range(1,nsteps+1),
                   'Average squared distance': avg_sqdist})

#fit linear model to the average squared distance using numpy with an intercept of 0
x = np.array([i for i in range(1,nsteps+1)])
y = avg_sqdist
x = x[:,np.newaxis] #change to column index
a, _, _, _ = np.linalg.lstsq(x, y,rcond=-1) #fit linear model with zero intercept and machine precision tolerance
df['Trend'] = a*x

#plot the average squared distance from the origin
fig, ax = plt.subplots()
ax.plot(df['Number of steps'], df['Average squared distance'], label='Average Squared Distance')
ax.plot(df['Number of steps'], df['Trend'], linestyle='--',label='Linear Fit')
ax.set_xlabel('Number of steps'); ax.set_ylabel('Average squared distance')
ax.legend()
ax.set_title('Average Squared Distance from Origin')
st.pyplot(fig)
 
st.write('The code above is a simple example of a diffusion simulator. The user can select the number of random walks \
 in the simulation using the slider. The code generates a 3D random walk for the selected number of steps and plots \
    the walk in 2D.')
