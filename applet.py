import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title('Example Diffusion Simulator')

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
fig, ax = plt.subplots()
ax.plot(x_loc,z_loc)
ax.plot(x_loc[0],z_loc[0],'go')
ax.plot(x_loc[-1],z_loc[-1],'ro')
ax.set_xlabel('x position'); ax.set_ylabel('z position')
ax.title('3-D Random Walk plotted in 2D')
ax.legend(['Random Walk','Start','End'])
st.pyplot(fig)

dicta = {'Number of steps''Average squared distance': np.mean(sqdists,axis=0)}
df = pd.DataFrame(dicta)

#plot the average squared distance from the origin, averaged over all walks
st.line_chart(df)

