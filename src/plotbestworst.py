import sys, os
import sys
import torch
import numpy as np
import matplotlib.pylab as plt
pwd = os.getcwd()
sys.path.append('src') 
from load_data import *

def get_curves(input_folder,tce_numbers,planet_numbers):
    ## This gets all the training data that was a planet
    train_data, test_data = getData(input_folder)
    dataset = list(Data(test_data))
    planet_view = []
    for i in range(len(dataset)):
        if dataset[i][1][1] in tce_numbers:
            index = tce_numbers.index(dataset[i][1][1])
            if dataset[i][1][2] == planet_numbers[index]:
                planet_view.append(dataset[i][0][:201])
            
    return planet_view

def bestworst(IDs,input_folder):
    #Best Predictions
    best = IDs[:3,:]
    best = sorted(best, key=lambda x: x[0])
    best_tcenum = []
    best_planetnum = []
    for b in best:
        best_tcenum.append(b[0])
        best_planetnum.append(b[1])
    best_curves = get_curves(input_folder, best_tcenum,best_planetnum)
    
    #Worst Predictions
    worst = IDs[-5:, :]
    worst = sorted(worst, key=lambda x: x[0])
    worst_tcenum = []
    worst_planetnum = []
    for w in worst:
        worst_tcenum.append(w[0])
        worst_planetnum.append(w[1])
    worst_curves = get_curves(input_folder, worst_tcenum,worst_planetnum)

    fig, axs = plt.subplots(3, 1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
    x = np.arange(0,len(best_curves[0]),1)
    axs[0].set_title("Best Predictions")
    axs[0].scatter(x,best_curves[0],s=4)
    axs[0].set_xlabel("Label: "+str(best[0][2])+" Prediction: "+str(round(best[0][3],1))+" Error: "+str(round(best[0][4],1)))
    axs[1].scatter(x,best_curves[1],s=4)
    axs[1].set_xlabel("Label: "+str(best[1][2])+" Prediction: "+str(round(best[1][3],1))+" Error: "+str(round(best[1][4],1)))
    axs[2].scatter(x,best_curves[2],s=4)
    axs[2].set_xlabel("Label: "+str(best[2][2])+" Prediction: "+str(round(best[2][3],1))+" Error: "+str(round(best[2][4],1)))
    plt.show()
    
    fig, axs = plt.subplots(3, 1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
    axs[0].set_title("Worst Predictions")
    axs[0].scatter(x,worst_curves[0],s=4)
    axs[0].set_xlabel("Label: "+str(worst[0][2])+" Prediction: "+str(round(worst[0][3],1))+" Error: "+str(round(worst[0][4],1)))
    axs[1].scatter(x,worst_curves[1],s=4)
    axs[1].set_xlabel("Label: "+str(worst[1][2])+" Prediction: "+str(round(worst[1][3],1))+" Error: "+str(round(worst[1][4],1)))
    axs[2].scatter(x,worst_curves[2],s=4)
    axs[2].set_xlabel("Label: "+str(worst[2][2])+" Prediction: "+str(round(worst[2][3],1))+" Error: "+str(round(worst[2][4],1)))
    plt.show()
    
    