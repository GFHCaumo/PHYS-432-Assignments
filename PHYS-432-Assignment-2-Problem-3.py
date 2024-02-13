#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:02:01 2024

@author: guilhermecaumo
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting timestep and total number of timesteps
dt = 5
Nsteps = 100

# Vortex rings initial conditions
y_v = np.array([-50, 50, -50, 50], dtype="f")  # y-positions of the 4 vortices
x_v = np.array([-100, -100, -175, -175], dtype="f")  # x-positions of the 4 vortices
k_v = np.array([40, -40, 40, -40])  # line vortex constants for the 4 vortices

# Set up the plot
plt.ion()
fig, ax = plt.subplots(1, 1)
p, = ax.plot(x_v, y_v, 'k+', markersize=10)  # You can adjust marker size and type

# Simulation grid
ngrid = 200
Y, X = np.mgrid[-ngrid:ngrid:360j, -ngrid:ngrid:360j]  # Adjust resolution as needed
vel_x = np.zeros(np.shape(Y))  # x-velocity
vel_y = np.zeros(np.shape(Y))  # y-velocity

# Masking radius
r_mask = 5

# Function to calculate velocity field
def calculate_velocity(vortex_x, vortex_y, k_v, X, Y, r_mask=0):
    r = np.sqrt((X - vortex_x)**2 + (Y - vortex_y)**2)
    
    r_masked = np.where(r < r_mask, np.nan, r)
    
    theta = np.arctan2(Y - vortex_y, X - vortex_x)
    
    vel_x = k_v * np.sin(theta) / r_masked
    vel_y = -k_v * np.cos(theta) / r_masked
    
    return vel_x, vel_y

# Initial velocity field calculation
for i in range(len(x_v)):
    vx, vy = calculate_velocity(x_v[i], y_v[i], k_v[i], X, Y, r_mask)
    vel_x += vx
    vel_y += vy

ax.set_xlim([-ngrid, ngrid])
ax.set_ylim([-ngrid, ngrid])
ax.streamplot(X, Y, vel_x, vel_y, density=[1, 1])
fig.canvas.draw()

# Evolution
count = 0
while count < Nsteps:
    # Compute and update advection velocity
    new_vel_x = np.zeros_like(x_v) # Corrected: Holds x-velocity for the vortices
    new_vel_y = np.zeros_like(y_v) # Corrected: Holds y-velocity for the vortices
    
    for i in range(len(x_v)):
        for j in range(len(x_v)):
            if i != j:
                r = np.sqrt((x_v[j] - x_v[i])**2 + (y_v[j] - y_v[i])**2)
                new_vel_x[i] -= k_v[j] * (y_v[j] - y_v[i]) / r**2
                new_vel_y[i] += k_v[j] * (x_v[j] - x_v[i]) / r**2
        
    # Update the positions of vortices
    x_v += new_vel_x * dt
    y_v += new_vel_y * dt
    
    # Re-initialize the total velocity field for the grid
    vel_x = np.zeros(np.shape(Y))
    vel_y = np.zeros(np.shape(Y))

    for i in range(len(x_v)):
        vx, vy = calculate_velocity(x_v[i], y_v[i], k_v[i], X, Y, r_mask)
        vel_x += vx
        vel_y += vy
    
    # Update plot
    for collection in ax.collections:
        collection.remove()
    for patch in ax.patches:
        patch.remove()

    p.set_xdata(x_v)
    p.set_ydata(y_v)

    ax.streamplot(X, Y, vel_x, vel_y, density=[1, 1])

    fig.canvas.draw()
    plt.pause(0.001) # Play around with the delay time for better visualization
    count += 1

plt.ioff() # Turn off interactive mode to prevent further updates
plt.show()
