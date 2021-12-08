import sys, os
import numpy as np
import matplotlib.pylab as plt

pwd = os.getcwd()
sys.path.append('src') 
from load_data import *


def get_planets(input_folder):
    '''
    get_planets gets all of the planet light curve data
    input_folder : path to the input data folder 
    '''

    train_data, test_data = getData(input_folder)
    train_dataset = list(Data(train_data))
    planet_view = []

    for i in range(len(train_dataset)):
        if train_dataset[i][1][1] == 5956342 and train_dataset[i][1][2]==4.0:
            # Changed from [i][1] to [i][1][0] for the new list format of the labels
            planet_view.append(train_dataset[i][0])
        elif train_dataset[i][1][1] == 3858919 and train_dataset[i][1][2]==2.0:
            planet_view.append(train_dataset[i][0])
            
    return planet_view
    

def create_heatmap(model, input_folder,view, filename):
    planet_tce = get_data(input_folder)
    
    # Index =1 is for planet, Index=0 is for binaries
    index = 1
    if view=='local':
        chosen_spectrum = planet_tce[index][:201]
    elif view == 'global':
        chosen_spectrum = planet_tce[index][201:]
    else:
        chosen_spectrum = planet_tce[index]

    # Create a window that will move along the points and turn those points to zero
    window = np.arange(-49, 1, 1)
    ones = np.ones(50)
    
    windows = []
    probabilities = []
    
    #The window will move as long as the first element of window is less than the last element of the spectrum
    while window[0]<=len(planet_tce[index]):
        #The new spectrum that will have certain values become zero
        new_spectrum = torch.tensor([i for i in planet_tce[index]])
        
        # The positions that are seen in window will become zero 
        for i in window:
            if i >= 0 and i < len(new_spectrum):
                new_spectrum[int(i)] = 0.0
                
        # Probability of the new spectrum being a planet
        new_data = new_spectrum.view(1, -1)
        prob = model(new_data)
        
        # Add this specific window and its probability to a list
        windows.append(window)
        probabilities.append(prob)
        
        #Move the window
        window = window+ones
        
    if view=='global':
        heats = []
        for i in range(201,201+len(chosen_spectrum)):
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
    else:
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
    plt.scatter(x, chosen_spectrum, c=heats, s=20, cmap="winter", edgecolor='black', linewidth=1)
    plt.colorbar(label="Predicted Planet Probability")
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False, 
    left=False,# ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
    plt.savefig(filename)
 
