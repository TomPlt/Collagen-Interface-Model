# -*- coding: utf-8 -*-

import matplotlib
from matplotlib import pyplot as plt 
import numpy as np 
import time
from matplotlib.colors import LogNorm


# helper functions 

def notInitialSphere(a, r, array):
    """
    function which collects all the values of an array outside an 
    initally defined sphere
    
    parameters:
        a (int): dimension of the system
        r (int): radius of initial spheroid
        array (np.array): array which is evaluated 
        
    returns:
        notspere(list): list of cell density values which are not within 
        the initial spheroid
    """
        
    notsphere = []
    for x in np.arange(1, a+1):
        for y in range(1, a+1):          
            x1 = x - (a+1)/2
            y1 = y - (a+1)/2
            if x1*x1 + y1*y1 >= (r+1)**2: 
                notsphere.append(array[x,y,-2])
    return notsphere
    


def pyramid_outside(a, r, array):
    '''
    function which collects all the values of an array outside an initally 
    defined sphere and within a pyramid shape
    
    parameters: 
        a (int): dimension of the system
        r (int): radius of initial spheroid
        array (np.array): array which is evaluated 
    
    return:
        pyramid(list): list of cell density values which are not within 
        the initial spheroid and within a pyramid"""
        
    '''
    
    pyramid = []
    for x in np.arange((a+1)/2, a+1):

        for y in np.arange(a-(x+1) , x+1): 
     
            x1 = x - (a+1)/2
            y1 = y - (a+1)/2
           
            if x1*x1 + y1*y1 >= (r+1)**2:
                pyramid.append(array[int(x),int(y),-2])

    return pyramid 


def cellGrid(a, r, cell_dens, ecm_inter, ecm_3D):  
    """
    function which generates the inital grid of cell and 
    ECM densities of the sandwich model
    
    parameters: 
        a (int): dimension of the system
        r (int): radius of initial spheroid
        array (np.array): array which is evaluated 
        cell_dens (float): inital spheriod cell density
        ecm_inter (float): ecm density within the interface
        ecm_3d (float): ecm density in the 3d region 
            
    return: 
        grid (np.array): spatial array of cell and ecm densities 
        
    """

    grid  = np.zeros((a+2,a+2, 2))

    for x in np.arange(1, a+1):
        for y in np.arange(1, a+1):
            x1 = x - (a+1)/2
            y1 = y - (a+1)/2
            
            
            if x1*x1 + y1*y1 < r*r:
                    grid[x, y, :] = np.array([cell_dens, 0])
                
            if x1*x1 + y1*y1 == r*r and x == (a+1)/2:
                    grid[x, y, :] = np.array([cell_dens, ecm_inter])
                
                
            if x1*x1 + y1*y1 == r*r and x != (a+1)/2:
                    grid[x, y, :] = np.array([cell_dens, ecm_3D])                
                    
                    
            if x1*x1 + y1*y1 > r*r and x == (a+1)/2: 
                    grid[x, y, :]  = np.array([0, ecm_inter])
                    
                            
            if x1*x1 + y1*y1 > r*r and x != (a+1)/2:
                    grid[x, y, :] = np.array([0, ecm_3D])

      
    return grid         

def periodic_rb(a, array):
    '''
    function which defines the von-Neumann neighborhood
    
    parameters: 
         a (int): dimension of the system
         array (np.array): array to which pb are to be applied
         
    '''
    
    array[0, ...] = array[a, ...]
    array[a+1, ...] = array[1, ...]
    array[:, 0, ...] =  array[:, a, ...]
    array[:, a+1, ...] =  array[:, 1, ...]
    
    
    

def neighbor_values(a, array):
    '''
    function which returns an array at each point in space representing 
    the values in the corresponding von-Neumann neighbors
    
    parameters: 
        a (int): dimension of the system
        array (np.array): array to which pb are to be applied
    
    return:
        neigbors (array): array of neighboring values
        
    '''
    periodic_rb(a, array)
    neighbors = np.zeros((a+2, a+2, 4, 2))
    for x in np.arange(1, a+1):
        for y in np.arange(1, a+1):
            neighbors[x, y, 0] = array[x+1, y, -2:] 
            neighbors[x, y, 1] = array[x-1, y, -2:] 
            neighbors[x, y, 2] = array[x, y+1, -2:]     
            neighbors[x, y, 3] = array[x, y-1, -2:] 
           
    return neighbors 


def updateGrid(a, r, cell_dens, ecm_inter, ecm_3d, d1, m, t, f_c):
    '''
    main function which updates an initial configuration of ECM and 
    cell densities for a specified number of timesteps
    
    parameters: 
        a (int): dimension of the system
        r (int): radius of initial spheroid
        cell_dens (float): inital spheriod cell density
        ecm_inter (float): ecm density within the interface
        ecm_3d (float): ecm density in the 3d region 
        d1 (float): degradation constant
        m (float): growth rate
        t (int): timesteps:
        f_c(float): critical ecm density 
    ''' 
    a = a + 1
    h = 2*a + 1


    d2 = d1 * 0.1

    tau = 48     

    
    Grid = cellGrid(a, r, cell_dens, ecm_inter, ecm_3d)
   
    
    neighbors = neighbor_values(a, Grid)   
    
    ecm = Grid[1:-1, 1:-1 ,-1] 
    ecm_neigh = neighbors[..., -1]
    ecm_neigh_sum = np.sum(ecm_neigh, axis=-1)
    
    
    cd_neigh = neighbors[..., -2]
    cd = Grid[1:-1, 1:-1, -2]
    cd_neigh_sum = np.sum(cd_neigh, axis=-1)
   
     
     
    start_time = time.time()
    times = range(t)
    for t in times: 
        
        neighbors = neighbor_values(a, Grid)        

        ecm = Grid[1:-1, 1:-1 ,-1] 
        cd = Grid[1:-1, 1:-1, -2]   
   
        ecm_neigh = neighbors[..., -1]
        ecm_neigh_exp_sum = np.sum(np.exp(-ecm_neigh[1:-1, 1:-1]/f_c), axis=-1)
        ecm_neigh_sum = np.sum(ecm_neigh[1:-1, 1:-1], axis=-1)
    
        
        
        cd_neigh = neighbors[..., -2]
        cd_neigh_sum = np.sum(cd_neigh[1:-1, 1:-1], axis=-1)
        ecm2 = np.exp(-ecm/f_c)
        
      
        new_ecm = ecm - d1 * np.multiply(cd, ecm) - d2 *np.multiply(cd_neigh_sum, ecm)
        
        new_cd = np.multiply(2**(1/tau), cd) + m*np.multiply(np.multiply(cd_neigh_sum, ecm_neigh_sum), ecm2) - m*np.multiply(np.multiply(ecm_neigh_exp_sum, ecm_neigh_sum), cd)
        
        new_cd[new_cd>1] = 1
        Grid[1:-1, 1:-1, -2] = new_cd  
        Grid[1:-1, 1:-1, -1] = new_ecm 
   

        
    outsideSphere = notInitialSphere(a, r, Grid)
    pyramid = pyramid_outside(a, r, Grid)

    
    ratio_of_interface_mig = round(((sum(outsideSphere)
                                     -4*sum(pyramid))/sum(outsideSphere)),3)

    
    # ignoring all values which are smaller than 10**-5 for the plots 
    
    Grid[1:-1, 1:-1, -2] [Grid[1:-1, 1:-1, -2] < 10**(-5)] = 10**(-5)
    
    
    
    
    
    
    # print the calculated percentage of interface migration 

    print("the ratio between interface migration and total migration = ",
          ratio_of_interface_mig)
    

    
    # plots with matplolib
    
    f = plt.figure(figsize=(11,4))
    
    ax1 = f.add_subplot(1,2,1)
    ax1.title.set_text('ECM Density')
    cmap = matplotlib.cm.viridis_r
    plt.imshow(Grid[1:-1, 1:-1, -1], cmap = cmap, vmin=0, vmax=1)
    plt.colorbar()
    
    ax2 = f.add_subplot(1,2,2)
    ax2.title.set_text('Cell Density')

    cmap = matplotlib.cm.viridis  
    plt.imshow(Grid[1:-1, 1:-1, -2], cmap=cmap, norm= LogNorm())
    plt.colorbar()
    
    

    plt.show()   

      

 
if __name__ == "__main__":
    a = 80 
    r = 20 
    cell_dens = 1
    ecm_inter = 0.26
    ecm_3d = 0.5 
    d1 = 0 
    m = 0.2
    t = 100 
    f_c = 0.1
    updateGrid(a=a, r=r, cell_dens=cell_dens, ecm_inter=ecm_inter
               , ecm_3d=ecm_3d, d1=d1, m=m, t=t, f_c=f_c)
