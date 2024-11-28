from PIL import Image
from pylab import *

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
from numpy import interp

from math import tau
from scipy.integrate import quad


def create_close_loop(image_name, level=[200]):
    # Prepare Plot
    ## sebs garbage code: 
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'datalim')

    # read image to array, then get image border with contour
    im = array(Image.open(image_name).convert('L'))
    
    ## sebs garbage code: 
    contour_plot = ax.contour(im, levels=level, colors='black', origin='image')

    # Get Contour Path and create lookup-table
    contour_path = contour_plot.collections[0].get_paths()[0]
    x_table, y_table = contour_path.vertices[:, 0], contour_path.vertices[:, 1]
    time_table = np.linspace(0, tau, len(x_table))

    # Simple method to center the image
    x_table = x_table - min(x_table)
    y_table = y_table - min(y_table)
    x_table = x_table - max(x_table) / 2
    y_table = y_table - max(y_table) / 2
    # Visualization
    ax.clear() #clear previous non centred version
    ax.plot(x_table,y_table,'k-') # display centred version of image
    

    return time_table, x_table, y_table

def determine_timesteps(timesteps):
    dt = np.linspace(0, tau, timesteps)
    return dt

#seb wants to revisit this bit
def coordinate_to_complex_number(t, time_table, x_table, y_table):
    x_interp = interp(t, time_table, x_table)
    y_interp = interp(t, time_table, y_table)
    coordinate_as_complex_vector =  x_interp + 1j*y_interp
    return coordinate_as_complex_vector

## THEN REWRITE THIS BIT 
def DFT(t, coef_list, order=5):
    #kernel is the list of all of the e terms
    #series is the coefficients multiplied by all the e terms 
    kernel = np.array([np.exp(-n*1j*t) for n in range(-order, order+1)]) 
    series = np.sum((coef_list[:,0]+1j*coef_list[:,1]) * kernel[:])
    return np.real(series), np.imag(series)

## THIS IS THE NEXT BIT TO REWRITE
def coef_list(time_table, x_table, y_table, order=5):
    coef_list = []
    harmonics = range(-order, order+1) 
    for harmonic in harmonics:
    
        # "lambda t:" is essentially a variable declaration, where it declares the bit that follows as a function
        # that can be passed into the quad function.
        # the quad takes a function and integrates it between bounds - in this instance, 0 and tau
        real_coef = quad(lambda t: np.real(coordinate_to_complex_number(t, time_table, x_table, y_table) * np.exp(-harmonic*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau # the [0] means it only returns a float value from the quad function, divided by tau to normalise 
        imag_coef = quad(lambda t: np.imag(coordinate_to_complex_number(t, time_table, x_table, y_table) * np.exp(-harmonic*1j*t)), 0, tau, limit=100, full_output=1)[0]/tau
        #magnitude = sqrt(real_coef**2+imag_coef**2)
        coef_list.append([real_coef, imag_coef])
        
        # the result of this is our an and bn terms but they are numerical values always, not functions.
        #we can then put these terms back into our Fourier synthesis equation which is our DFT function
    return np.array(coef_list)

#def sebs_garbage_coef_list(time_table, x_table, y_table, order=15):
#    coef_list = []

#        equation = coordinate_to_complex_number(t, time_table, x_table, y_table) * np.exp(-harmonic*1j*t)

def sort_key(coef):
    
    return coef[2]

def visualize(x_DFT, y_DFT, coef, order, space, fig_lim):
    fig, ax = plt.subplots()
    lim = max(fig_lim)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')

    # Initialize
    line = plt.plot([], [], 'k-', linewidth=2)[0]
    radius = [plt.plot([], [], 'r-', linewidth=0.5, marker='o', markersize=1)[0] for _ in range(2 * order + 1)]
    circles = [plt.plot([], [], 'r-', linewidth=0.5)[0] for _ in range(2 * order + 1)]

    def update_c(c, t):
        new_c = []
        for i, j in enumerate(range(-order, order + 1)):
            dtheta = -j * t
            ct, st = np.cos(dtheta), np.sin(dtheta)
            v = [ct * c[i][0] - st * c[i][1], st * c[i][0] + ct * c[i][1]]
            new_c.append(v)
        return np.array(new_c)

    def sort_velocity(order):
        idx = []
        for i in range(1,order+1):
            idx.extend([order+i, order-i]) 
        return idx    
    
    def animate(i):
        # animate lines
        line.set_data(x_DFT[:i], y_DFT[:i])
        # animate circles
        r = [np.linalg.norm(coef[j]) for j in range(len(coef))]
        pos = coef[order]
        c = update_c(coef, i / len(space) * tau)
        idx = sort_velocity(order)
        for j, rad, circle in zip(idx,radius,circles):
            new_pos = pos + c[j]
            rad.set_data([pos[0], new_pos[0]], [pos[1], new_pos[1]])
            theta = np.linspace(0, tau, 50)
            x, y = r[j] * np.cos(theta) + pos[0], r[j] * np.sin(theta) + pos[1]
            circle.set_data(x, y)
            pos = new_pos
                
    # Animation
    ani = animation.FuncAnimation(fig, animate, frames=len(space), interval=5)
    return ani
