#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The code below is the submission for PHYS 432 - Assignment 4 - Problem 3,
here the code simulates the evolution of a strong adiabatic shock of 
adiabatic index 5/3 with reflective boundary conditions on both sides. 
The code benefitted from advice from the collaborators identified below 
and notes from lecture notes for PHYS 432, particularly the code 
provided by Prof. Andrew Cumming was used below and adapted to address 
the problem based on the provided steps in the assignment. Sections of the code
altered for the assignment have comments explaining functionality below.

@author: Guilherme Caumo
@collaborators: Maryn Askew
"""

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib 

def advect_step(q,u,dt,dx):
    
    # calculate fluxes
    J = np.zeros(n-1)
    
    for i in range(n-1):
        if u[i]>0.0:
            J[i] = q[i]*u[i]
        else:
            J[i] = q[i+1]*u[i]
            
    # now do the update
    q[1:-1] = q[1:-1] - (dt/dx)*(J[1:]-J[:-1])
    q[0] = q[0] - (dt/dx)*J[0]
    q[-1] = q[-1] + (dt/dx)*J[-1]
    
    return q

nsteps = 1900  # number of steps
alpha = 0.1    # timestep
gamma = 5/3    # adiabatic index

n = 1900
#cs2 = 1.0
x = np.linspace(0, 1, n)
dx = x[1]-x[0]
dt = 0.0001
q1 = np.ones(n)  # includes initial conditions q1(t=0)=1
q2 = np.zeros(n) # q2(t=0)=0

# Additional q variable added for the exercise
q3 = np.ones(n)  

#Here we define pressure and speed of sound
P = (gamma - 1) * (q3 - 0.5 * q2 ** 2 / q1)
cs2 = gamma * P / q1


#%% Initial conditions

AA = 60.0  # amplitude

# Gaussian perturbation
q3 = 1.0 + AA * np.exp(-(x-0.5)**2/0.0025)

# Initial Mach number
mach = q2 / q1 / np.sqrt(cs2)

#%% Set up the plots
plt.ion()

plt.subplot(211)
x1, = plt.plot(x,q1,'b-',ms=1)
plt.axhline(y=1/4, ls = '--', color = 'black', label='Adiabatic shock (M1>>1)')
plt.xlim([0,1])
plt.ylim([0,5])
plt.ylabel('Density')
plt.legend(loc='upper right')

plt.subplot(212)
x2, = plt.plot(x,mach,'r-',ms=1)
plt.ylim([-3,3])
plt.ylabel('Mach number')
plt.xlabel('Position (x)')

plt.draw()

# now do the iterations
count = 0 

while count < nsteps:

    #Following the steps 1-9 detailed in the instructions:
    # 1. Compute the advection velocity by averaging adjacent grid point velocities
    u = 0.5*((q2[:-1]/q1[:-1])+(q2[1:]/q1[1:]))
    
    # 2. Advect density and momentum using the calculated velocities
    q1 = advect_step(q1,u,dt,dx) # Advect density
    q2 = advect_step(q2,u,dt,dx) # Advect momentum
    
    # 3. and 4. Update momentum with pressure source term, based on ideal gas law
    P = (gamma - 1) * (q3 - 0.5 * q2 ** 2 / q1)
    q2[1:-1] = q2[1:-1] - dt * (P[2:]-P[:-2])/(2.0*dx)
    q2[0] = q2[0] - dt * (P[1]-P[0])/(2*dx)
    q2[-1] = q2[-1] - dt * (P[-1]-P[-2])/(2*dx)
    
    # 5. Re-calculate the advection velocities after momentum update
    u = 0.5*((q2[:-1]/q1[:-1])+(q2[1:]/q1[1:]))
    
    # 6. Advect energy with updated velocities
    q3 = advect_step(q3,u,dt,dx)
    
    # 7. and 8. Update energy with pressure work and kinetic energy changes
    uu = q2/q1 # Velocity
    P = (gamma - 1) * (q3 - 0.5 * q2 ** 2 / q1)
    q3[1:-1] = q3[1:-1] - dt * (P[2:]*uu[2:]-P[:-2]*uu[:-2])/(2.0*dx)
    q3[0] = q3[0] - dt * (P[1]*uu[1]-P[0]*uu[0])/(2*dx)
    q3[-1] = q3[-1] - dt * (P[-1]*uu[-1]-P[-2]*uu[-2])/(2*dx)
    
    # 9. Recalculate pressure and sound speed for next iteration
    P = (gamma - 1) * (q3 - 0.5 * q2 ** 2 / q1)
    cs2 = gamma * P / q1
    
    # Update Mach number to reflect changes in flow speed relative to sound speed
    mach = q2 / q1 / np.sqrt(cs2)
    
    # update the plot
    x1.set_ydata(q1)
    x2.set_ydata(mach)
    plt.draw()
    plt.pause(0.001)
    
    count +=1







