import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Example Diffusion Simulator')
st.write('This is a simple example of a diffusion simulator. The user can \
select the number of steps in the simulation using the slider below.')

val = st.slider('Select a value', 20, 200, 100)

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
fig, ax = plt.subplots(2)
ax[0].plot(x_loc,z_loc)
ax[0].plot(x_loc[0],z_loc[0],'go')
ax[0].plot(x_loc[-1],z_loc[-1],'ro')
ax[0].set_xlabel('x position'); ax[0].set_ylabel('z position')
ax[0].set_title('3-D Random Walk plotted in 2D')
ax[0].legend(['Random Walk','Start','End'])
#plot the squared distance from the origin 
avesqdists = np.mean(sqdists,axis=0)
ax[1].plot(avesqdists,label="Average squared distance, nwalks = 100")
ax[1].legend()
ax[1].set_xlabel("Number of steps"); ax[1].set_ylabel("Average squared distance from origin")
st.pyplot(fig)

num_steps = np.linspace(0,len(avesqdists),len(avesqdists))
dicta = {'Number of steps':num_steps,'Average squared distance': avesqdists}
df = pd.DataFrame(dicta)

#plot the average squared distance from the origin, averaged over all walks
fix, ax = plt.subplots()
sns.lineplot(data=df, x='Number of steps', y='Average squared distance', ax=ax)

