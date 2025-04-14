# -*- coding: utf-8 -*-
"""
Training NN for CEST quantification - first with the full acquisition schedule (no optimization)

We use a simple fully-connected NN for concentration predictions

Hyperparameters were found through separate optimization through dictionnary-testing process

Created on Wed Jul 24 9:29:12 2022

@author: Cecile Maguin
"""

# In[1]: Imports

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import os

from NN_quantif_tools.NN_subfunctions import reconstruction_network, train_model

current_seed = 136
th.manual_seed(current_seed)

# In[2]: Load data and convert to tensors


# Load data
DataName = "Data/Dataset_GluGlc_2025-04-13/"
Data = np.load(DataName + "Data.npy")
Ground_truth = np.load(DataName + "Ground_truth_for_training.npy")

N_data = np.size(Data, 0)

from NN_quantif_tools.Data_generation_utils import read_info

Nacq, B1_list, tsat_list, Offsets_list = read_info(DataName + "Dataset_info.txt")


# Prepare training and validation set
Training_set_size = round(0.8 * N_data)  # 80% for training
shuffling = th.randperm(N_data)  # shuffle datasets
Data = Data[shuffling, :]
Ground_truth = Ground_truth[shuffling, :]
Training_set = th.FloatTensor(Data[0:Training_set_size, :])
Ground_truth_training = th.FloatTensor(Ground_truth[0:Training_set_size, :])
Validation_set = th.FloatTensor(Data[Training_set_size:N_data, :])
Ground_truth_validation = th.FloatTensor(Ground_truth[Training_set_size:N_data, :])

Noutput = Ground_truth_training.size(1)

# In[3]: Create NN


NN_config = {
    "Layer_size": [300, 300],
    "Dropout_rate": [0, 0.1],
    "Output_size": Ground_truth_training.size(1),
    "Input_size": Nacq,
}

qCEST_NN = reconstruction_network(NN_config)

# In[5]: CPU or GPU attribution

if th.cuda.is_available():
    print("Working on GPU - Proceed to training \n")
else:
    input(
        "Working on CPU (you may be logged in to wrong station) - Proceed anyway ? \n"
    )
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

qCEST_NN.to(device)
Training_set = Training_set.to(device)
Ground_truth_training = Ground_truth_training.to(device)
Validation_set = Validation_set.to(device)
Ground_truth_validation = Ground_truth_validation.to(device)

# In[6]: Train NN for CEST quantification on training set data

Training_config = {
    "Learning_rate": 1e-3,
    "Weight_Decay": 1e-4,
    "Minibatch_Size": 200,
}

Nepochs = 15000

# Training
Loss_training, Error_validation, training_time = train_model(
    qCEST_NN,
    Training_set,
    Validation_set,
    Ground_truth_training,
    Ground_truth_validation,
    Training_config,
    Nepochs,
)

qCEST_NN.eval()

# In[7]: Training evaluation

# ----Plot training loss and validation----
f1 = plt.figure()
# Plot the loss function
plt.plot(Loss_training)
plt.title("Loss function evolution during training sequence")
plt.xlabel("Number of epochs")
plt.ylabel("Loss function")
plt.yscale("log")
plt.show()
f1.savefig("outputs/Loss_during_training.svg", format="svg", dpi=660)

Var = ["Glu", "Glc"]
f2 = plt.figure()
for i in range(Noutput):
    plt.plot(np.sqrt(Error_validation[:, i]))
    plt.title("RMSE on validation set during training sequence")
    plt.xlabel("Number of epochs")
    plt.ylabel("Error on [" + Var[i] + "] estimation")
    plt.show()
f2.savefig("outputs/Validation_error_during_training.svg", format="svg", dpi=660)

# ----Get an idea of NN performance----
# test on 250 samples
samples = range(250)
prediction = qCEST_NN.forward(Validation_set[samples, :])
prediction = prediction.cpu().detach().numpy()
# for [Glu]
Glu_prediction = prediction[:, 0]
GT_test = Ground_truth_validation[samples, 0].cpu().detach().numpy()
Error_on_Glu = np.square((Glu_prediction - GT_test))
print(
    "RMSE on [Glu] prediction = ",
    format(np.sqrt(np.mean(Error_on_Glu)), ".2f"),
    " mM (std =",
    format(np.sqrt(np.std(Error_on_Glu)), ".2f"),
    ")",
)
# for [Glc]
Glc_prediction = prediction[:, 1]
GT_test = Ground_truth_validation[samples, 1].cpu().detach().numpy()
Error_on_Glc = np.square((Glc_prediction - GT_test))
print(
    "RMSE on [Glc] prediction = ",
    format(np.sqrt(np.mean(Error_on_Glc)), ".2f"),
    " mM (std =",
    format(np.sqrt(np.std(Error_on_Glc)), ".2f"),
    ")",
)

# In[8]: Save model

savepath = "outputs/"
# Choose name
Name = savepath + "Model_GluGlc_" + str(date.today()) + ".npy"
flag = 0
if os.path.exists(Name):
    ans = input("Probably overwriting previous model - ok with that ? (Y) \n")
# Here we create a dictionnary to store what we need:
# 1/ the epoch number
# 2/ the paramters of the model
# 3/ the paramters of the optimizer
# 4/ the loss function we used
tosave_data = {
    "model_config": NN_config,
    "model_state_dict": qCEST_NN.state_dict(),
    "loss": Loss_training,
    "error_validation": Error_validation,
    "epoch": Nepochs,
    "timing": training_time,
    "Nacq": Nacq,
    "B1_list": B1_list,
    "tsat_list": tsat_list,
    "Offsets_list": Offsets_list,
    "random seed": current_seed,
    "Data_set_used_for_training": DataName,
    "date": date.today(),
}
# Write a pickle file using pytorch
th.save(tosave_data, Name)

# In[9]: Validation on real experimental data

filepath = "/Data/Fantomes_21-03-2022/Averaged_Zspectrum/Manual_B0_correction/"

from scipy.io import loadmat
import os

files = os.listdir(filepath)

if files[0] == "Manual_B0_correction":
    files = files[1 : len(files)]
# Nexp=len(files)
selected_files = [1, 2, 3, 4, 5]
Nexp = 10

# Load offset ppm
xZspec_exp = loadmat(filepath + "freq_ppm.mat")
xZspec_exp = np.squeeze(xZspec_exp["freq_ppm"])
xZspec_exp = xZspec_exp[1 : len(xZspec_exp) - 1]
B1s_exp = loadmat(filepath + "B1values.mat")
B1s_exp = np.squeeze(B1s_exp["B1values"])
tsats_exp = loadmat(filepath + "tsatvalues.mat")
tsats_exp = np.squeeze(tsats_exp["tsat"])

graph, (plot1, plot2) = plt.subplots(1, 2)

B1s = np.array([3, 5, 7, 3, 5, 4, 6])

tsats = np.array([1, 1, 1, 4, 4, 1, 1])

# load experimental Zspectra data
Data_phantoms = np.zeros((Nexp, np.size(B1s_exp), len(xZspec_exp)))
MTR_phantoms = np.zeros((Nexp, np.size(B1s_exp), len(xZspec_exp)))
Data_phantoms_NN = np.zeros((Nexp, Nacq))
AllZspec = np.zeros((Nexp, np.size(B1s_exp), len(xZspec_exp)))
index_data_NN = np.zeros(len(B1s_exp), dtype=int)

for zz in range(Nexp):
    Zavg = loadmat(os.path.join(filepath, "Zavg_tube" + str(zz + 1) + "_corr"))
    Zavg = Zavg["Z_avg_corr"]
    Zavg = np.array(Zavg)

    # Interpolate data to remove NaN features
    for iB1 in range(np.size(B1s_exp)):
        Zspec = np.squeeze(Zavg[iB1, :])

        nan_values = np.argwhere(np.isnan(Zspec))
        unreliable_values = np.isnan(Zspec)
        for nan_index in nan_values:
            Zspec[nan_index] = np.interp(
                xZspec_exp[nan_index],
                xZspec_exp[~unreliable_values],
                Zspec[~unreliable_values],
            )
        if zz == 0:
            if np.any(np.logical_and(B1s == B1s_exp[iB1], tsats == tsats_exp[iB1])):
                index_data_NN[
                    int(
                        np.argwhere(
                            np.logical_and(B1s == B1s_exp[iB1], tsats == tsats_exp[iB1])
                        )
                    )
                ] = iB1
        AllZspec[zz, iB1, :] = Zspec
Data_phantoms = AllZspec[:, index_data_NN, :]

# Reformat in input shape for neural network
Data_phantoms_NN = np.reshape(Data_phantoms, (Nexp, -1))
Data_phantoms_NN = th.FloatTensor(Data_phantoms_NN)

# Load expected experimental concentrations
Expected_Glc = loadmat(filepath + "Expected_Glc.mat")
Expected_Glc = np.squeeze(Expected_Glc["Glc"])

Expected_Glu = loadmat(filepath + "Expected_Glu.mat")
Expected_Glu = np.squeeze(Expected_Glu["Glu"])

Expected_pH = loadmat(filepath + "Expected_pH.mat")
Expected_pH = np.squeeze(Expected_pH["pH"])


# Test out NN prediction performance
prediction_phantoms = qCEST_NN.forward(Data_phantoms_NN)
prediction_phantoms = prediction_phantoms.detach().numpy()

# For [Glc]
Glc_prediction_phantoms = prediction_phantoms[:, 0]
# print(Glc_prediction_phantoms)
Error_on_Glc_phantoms = np.square(Glc_prediction_phantoms - Expected_Glc)
print(np.sqrt(np.mean(Error_on_Glc_phantoms)))
plt.figure()
plt.plot(Expected_Glc, Expected_Glc, "k")
plt.plot(Expected_Glc, Glc_prediction_phantoms, "b+")
plt.xlabel("True [Glc] (mM)")
plt.ylabel("Predicted [Glc] (mM)")

# For [Glu]
Glu_prediction_phantoms = prediction_phantoms[:, 1]
Error_on_Glu_phantoms = np.square(Glu_prediction_phantoms - Expected_Glu)
print(np.sqrt(np.mean(Error_on_Glu_phantoms)))
plt.figure()
plt.plot(Expected_Glu, Expected_Glu, "k")
plt.plot(Expected_Glu, Glu_prediction_phantoms, "b+")
plt.xlabel("True [Glu] (mM)")
plt.ylabel("Predicted [Glu] (mM)")
