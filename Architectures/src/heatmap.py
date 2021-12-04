import sys, os
import sys
import torch
import numpy as np
import matplotlib.pylab as plt
pwd = os.getcwd()
sys.path.append('src') 
from load_data import *
from linear_net import LinNet
from fc_net import FCNet

def get_data(input_folder):
    ## This gets all the training data that was a planet
    train_data, test_data = getData(input_folder)
    train_dataset = list(Data(train_data))
    planet_view = []
    for i in range(len(train_dataset)):
        if train_dataset[i][1][0] == 1.0:
            # Changed from [i][1] to [i][1][0] for the new list format of the labels
            planet_view.append(train_dataset[i][0])
            
    return planet_view
    

def create_heatmap(model,input_folder,view):
    planet_tce_global = get_data(input_folder)
    
    # Randomly chose one of the spectra for the heat map (choosing a view)
    index = 2
    if view=='local':
        chosen_spectrum = planet_tce_global[index][:201]
    elif view=='global':
        chosen_spectrum = planet_tce_global[index][201:]
    else:
        chosen_spectrum = planet_tce_global[index]

    #Creating a window that will move along the points and turn those points to zero
    window = np.arange(-49,1,1)
    ones = np.ones(50)
    
    windows = []
    probabilities = []
    
    #The window will move as long as the first element of window is less than the last element of the spectrum
    while window[0]<=len(chosen_spectrum):
        #The new spectrum that will have certain values become zero
        new_spectrum = torch.tensor([i for i in planet_tce_global[index]])
        
        # The positions that are seen in window will become zero 
        for i in window:
            if i>=0:
                new_spectrum[int(i)] = 0.0
                
        #Probability of the new spectrum being a planet
        prob = model(torch.reshape(new_spectrum,(1,len(new_spectrum))))
        
        #Add this specific window and its probability to a list
        windows.append(window)
        probabilities.append(prob)
        
        #Move the window
        window = window+ones
    
    heats = []
    for i in range(len(chosen_spectrum)):
        indices = []
        
        #If the index in chosen spectrum is in one of the windows, save that window index
        for w in range(len(windows)):
            if i in windows[w]:
                indices.append(w)
                
        #Average the probabilities of all the windows that had the chosen spectrum index
        heatsum = 0
        for j in indices:
            heatsum+=probabilities[j]
        heats.append(heatsum/len(indices))
    
    x = np.arange(0,len(chosen_spectrum),1)
    plt.scatter(x,chosen_spectrum,c=heats)
    plt.colorbar()
    plt.show()
 