#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code below is the submission for PHYS 432 - Assignment 2 - Problem 3,
here the code simulates the leapfrogging behaviour of vortices, using functions and loops
to calculate the velocity of the field across the grid and incremental velocities 
for each vortex due to the influences of the others. The code benefitted from advice from the
collaborators identified below. By specifications of the question the motion goes from left
to right and is animated when running the code through a terminal. The skeleton of the code
was given by Prof. Eve J. Lee via Crowdmark.

@author: Guilherme H. Caumo
@collab: Julien Hacot-Slonosky, Emilia Vlahos
February 12, 2024
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting timestep and total number of timesteps
dt = 5
Nsteps = 77 # Value chosen to ensure animation goes from left to right while both vortices are in frame

## Setting up initial conditions (vortex centres and circulation)
# Vortex rings
y_v = np.array([-50, 50, -50, 50], dtype="f")  # y-positions of the 4 vortices
x_v = np.array([-100, -100, -175, -175], dtype="f")  # x-positions of the 4 vortices
k_v = np.array([40, -40, 40, -40])  # line vortex constants for the 4 vortices

# Set up the plot
plt.ion()
fig, ax = plt.subplots(1, 1)
# mark the initial positions of vortices
p, = ax.plot(x_v, y_v, 'k+', markersize=10)  # You can adjust marker size and type

# draw the initial velocity streamline
# Simulation grid
ngrid = 200 # the dimension of your simulation grid
Y, X = np.mgrid[-ngrid:ngrid:360j, -ngrid:ngrid:360j]  # sets the resolution of the cartesian grid
vel_x = np.zeros(np.shape(Y))  # x-velocity
vel_y = np.zeros(np.shape(Y))  # y-velocity

# masking radius for better visualization of the vortex centres
r_mask = 5 # the radius of the mask around the vortex centres 

#within this mask, you will not plot any streamline 
#so that you can see more clearly the movement of the vortex centres

def calculate_velocity(vortex_x, vortex_y, k_v, X, Y, r_mask=0):
    """
    Function to calculate velocity field across the grid, 
    used for visualizing the flow around the vortices. 
    
    vortex_x : float or int, x-position of vortex
        
    vortex_y : float or int, x-position of vortex
        
    X : array, x-positions on the grid as outputted by ngrid
    
    Y : array, y-positions on the grid as outputted by ngrid
    
    r_mask : float or int, radius around the vortex relative to mask 
    
    Here azimuthal velocity, denoted k_v/r, is expressed as x and y components, with
    the cosine and sine of the angle being delta y/r and delta x/r, respectively.
    """
    # Calculates distance between points and vortex
    r = np.sqrt((X - vortex_x)**2 + (Y - vortex_y)**2)
    
    r_masked = np.where(r < r_mask, np.nan, r) # Mask values within radius
    
    theta = np.arctan2(Y - vortex_y, X - vortex_x)
    
    # Here we compute speeds
    vel_x = k_v * np.sin(theta) / r_masked
    vel_y = -k_v * np.cos(theta) / r_masked
    
    return vel_x, vel_y

# Initial velocity field calculation
for i in range(len(x_v)):
    vx, vy = calculate_velocity(x_v[i], y_v[i], k_v[i], X, Y, r_mask)
    vel_x += vx
    vel_y += vy
    
# set up the boundaries of the simulation box
ax.set_xlim([-ngrid, ngrid])
ax.set_ylim([-ngrid, ngrid])

# initial plot of the streamlines
ax.streamplot(X, Y, vel_x, vel_y, density=[1, 1])


fig.canvas.draw()

# Evolution
count = 0
while count < Nsteps:
    ## Compute and update advection velocity
    # Here we calculate new_vel_x and new_vel_y
    #These are incremental velocities for each vortex due to the influences of the others
    new_vel_x = np.zeros_like(x_v) # Initialize temporary arrays for new x velocities of vortices
    new_vel_y = np.zeros_like(y_v) # Initialize temporary arrays for new y velocities of vortices
    
    # Calculate new velocities for each vortex based on interactions with the others
    for i in range(len(x_v)):
        for j in range(len(x_v)):
            if i != j: # Ensure we don't calculate self-interaction
                # Distance between vortex i and j
                r = np.sqrt((x_v[j] - x_v[i])**2 + (y_v[j] - y_v[i])**2)
                # Update velocities due to vortex j on vortex i
                new_vel_x[i] -= k_v[j] * (y_v[j] - y_v[i]) / r**2
                new_vel_y[i] += k_v[j] * (x_v[j] - x_v[i]) / r**2
        
    # Update vortex positions based on the calculated velocities
    x_v += new_vel_x * dt
    y_v += new_vel_y * dt
    
    # Reinitialize velocity fields for the updated vortex positions
    vel_x = np.zeros(np.shape(Y))
    vel_y = np.zeros(np.shape(Y))

    for i in range(len(x_v)):
        vx, vy = calculate_velocity(x_v[i], y_v[i], k_v[i], X, Y, r_mask)
        vel_x += vx
        vel_y += vy
    
    ## update plot
    # the following two lines clear out the previous streamlines
    # Note: template code from problem was outdated for collection and patch, updated here
    for collection in ax.collections:
        collection.remove()
        
    for patch in ax.patches:
        patch.remove()

    p.set_xdata(x_v)
    p.set_ydata(y_v)

    ax.streamplot(X, Y, vel_x, vel_y, density=[1, 1])

    fig.canvas.draw()
    plt.pause(0.0001) # Play around with the delay time for better visualization
    count += 1

plt.ioff() # Turn off interactive mode to prevent further updates
plt.show()
