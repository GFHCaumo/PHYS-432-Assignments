#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code below is the submission for PHYS 432 - Assignment 3 - Problem 3,
here the code simulates the flow of basaltic lava. The code benefitted from advice from the
collaborators identified below and notes from lecture notes for 
PHYS 512 - Computational Physics with Applications.

@author: Guilherme Caumo
@collaborators: Julien Hacot-Slonosky

February 26, 2024
"""

#%% Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg  # Import scipy.linalg for solving tridiagonal systems

#%% Setup

# Set up the grid and diffusion parameters
n = 500  # Number of grid points

H = 1  # Height of lava layer in meters

dy = H/n  # Spatial step size

tf = 5  # Final time
nsteps = 500  # Number of time steps
dt = tf/nsteps  # Time step size

## Gravitational component for code
g = 9.81  # Gravitational acceleration in m/s^2
alpha = 10  # Slope inclination in degrees
rho = 2700  # Density of basaltic lava in kg/m^3 (from Wikipedia)
K = g*np.sin(np.deg2rad(alpha))  # Component of gravity along the slope

# Viscosity of basaltic lava (from Wikipedia), adjusted for density
v = 1e5/rho
v = 1  # Assuming a normalized viscosity for simplicity
beta = v * dt/dy**2  # Diffusion coefficient for the numerical scheme

height = np.linspace(0, H, n)  # Spatial grid
f1 = np.zeros(n)  # Initial speed of lava, initialized to zero

# Steady state solution for comparison purposes
u_steady = -(g/v)*np.sin(np.deg2rad(alpha))*(height**2/2 - H*height)

#%%% Plotting Setup

# Initialize interactive mode for live plot updates
plt.ion()
fig, ax = plt.subplots(1,1)

# Plotting objects for dynamic updates
pl, = ax.plot(height, f1, color="red", label = "Velocity of lava flow")  # Lava flow speed plot

# Reference plot for steady state solution
ax.plot(height, u_steady, "k", label="Steady state solution")

# Setting plot titles and labels
ax.set_title("Basaltic lava flow")
ax.set_ylabel("Speed of lava (m/s)")
ax.set_xlabel("Height of lava (m)")

plt.legend()

# Initial draw of the plot
fig.canvas.draw()

#%% Calculate the matrix A for the numerical scheme
#(based off of diffusion notes from PHYS 512 - Computational Physics with Applications)

# Define the coefficients for the tridiagonal matrix representing the discretized system
a = -beta * np.ones(n)  # Lower diagonal
b = (1+2*beta)*np.ones(n)  # Main diagonal
c = -beta * np.ones(n)  # Upper diagonal

# Modify coefficients for boundary conditions
a[0] = 0  # No-slip boundary condition at the bottom
b[-1] = 1 + beta  # Stress-free boundary condition at the top
c[-1] = 0

# Stack the diagonals to form the tridiagonal matrix A
A = np.row_stack((a, b, c))

#%% Time-stepping loop

for i in range(1, nsteps):
    # Solve the linear system for the next time step
    f1 = scipy.linalg.solve_banded((1, 1), A, f1 + dt * K)
    
    # Update the plot with the new speed data
    pl.set_ydata(f1)
    fig.canvas.draw()
    plt.pause(0.001)  # Short pause to allow for plot updates
    
# Keep the plot open after the simulation is done
plt.show()