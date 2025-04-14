# -*- coding: utf-8 -*-
"""
Feature selection scheme to select only the most relevant CEST acquisition points for our quantification problem

Example here for the Gluamate-Glucose problem, which also has specificity issue

Created on Wed Jul 27 12:58:13 2022

@author: Cecile Maguin
"""


# In[1]: Imports

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
from datetime import date

from NN_quantif_tools.NN_subfunctions import (
    reconstruction_network,
    train_model,
    feature_elimination,
)

current_seed = 136
th.manual_seed(current_seed)


# In[2]: Load previous NN model
Name = "outputs/Model_GluGlc_2025-04-13.npy"
Model_data = th.load(Name)

NN_config = Model_data["model_config"]

Error_validation = Model_data["error_validation"]
Loss_training = Model_data["loss"]
Nacq = Model_data["Nacq"]
B1_list = Model_data["B1_list"]
tsat_list = Model_data["tsat_list"]
Offsets_list = Model_data["Offsets_list"]

qCEST_NN = reconstruction_network(NN_config)

qCEST_NN.load_state_dict(Model_data["model_state_dict"])

# In[3]: Load training dataset, split and moves to GPU if available

# Load same data
X = th.FloatTensor(np.load(Model_data["Data_set_used_for_training"] + "/Data.npy"))
Y = th.FloatTensor(
    np.load(Model_data["Data_set_used_for_training"] + "Ground_truth_for_training.npy")
)

# Attribute GPU if available, CPU otherwise
if th.cuda.is_available():
    print("Working on GPU - Proceed to training \n")
else:
    input(
        "Working on CPU (you may be logged in to wrong station) - Proceed anyway ? \n"
    )
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

qCEST_NN.to(device)
X = X.to(device)
Y = Y.to(device)

# Take training set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=current_seed
)

# In[4]: Feature selection process : Greedy recursive feature elimination

# Copy NN model so as not overwrite
model = copy.deepcopy(qCEST_NN)
N_current_features = Nacq

# Number of features to remove per round
kj = 5
# End goal number of features
N_features_goal = 27

# Setup training during RFE
Training_config = {
    "Learning_rate": 1e-3,
    "Weight_Decay": 1e-4,
    "Minibatch_Size": 200,
}
Nepochs = 5000

remaining_features = list(range(N_current_features))
iteration = 0
Removed_list = []
Scores_of_removed_features = []
Model_performance = []

# Feature selection until we reach the desired number of features
while N_current_features > N_features_goal:

    print("----- Feature elimination round no " + str(iteration + 1) + " ----- \n")

    best_features, removed_features, scores_removed = feature_elimination(
        model, X_train[:, remaining_features], Y_train, N_current_features - kj
    )
    Removed_list.append(removed_features)
    Scores_of_removed_features.append(scores_removed)

    # for printing
    for j in range(kj):
        print(
            "Feature #"
            + str(removed_features[j])
            + " eliminated with score "
            + format(scores_removed[j], ".3f")
            + " (B1="
            + format(B1_list[removed_features[j]], ".0f")
            + "µT, tsat="
            + format(tsat_list[removed_features[j]], ".0f")
            + "s, offset="
            + format(Offsets_list[removed_features[j]], ".2f")
            + "ppm)  \n"
        )
    # Update number of features
    N_current_features += -kj
    # Update list of features
    remaining_features = [
        f for i, f in enumerate(remaining_features) if i not in removed_features
    ]

    # Retrain the model with reduced features
    print("Retraining the model \n")
    NN_config["Input_size"] = N_current_features
    model = reconstruction_network(NN_config)
    model.to(device)

    Loss_training, Error_validation, training_time = train_model(
        model,
        X_train[:, remaining_features],
        X_test[:, remaining_features],
        Y_train,
        Y_test,
        Training_config,
        Nepochs,
    )

    Model_performance.append(
        model.score(X_test[:, remaining_features], Y_test).cpu().detach().numpy()
    )

    iteration += 1
# In[5]: Results

f1 = plt.figure()
# Plot the loss function
plt.plot(np.sqrt(Model_performance))
plt.title("Model performance evolution during RFE")
plt.xlabel("Number of iteration")
plt.show()
f1.savefig("outputs/Performance_during_RFE.svg", format="svg", dpi=660)


# Save model trained on optimized features
tosave_data = {
    "model_config": NN_config,
    "model_state_dict": model.state_dict(),
    "loss": Loss_training,
    "error_validation": Error_validation,
    "epoch": Nepochs,
    "timing": model.perf_record["Training_time"],
    "Nacq": N_current_features,
    "B1_list": B1_list[remaining_features],
    "tsat_list": tsat_list[remaining_features],
    "Offsets_list": Offsets_list[remaining_features],
    "random seed": current_seed,
    "Data_set_used_for_training": Model_data["Data_set_used_for_training"],
    "date": date.today(),
}
# Write a pickle file using pytorch
th.save(tosave_data, "outputs/Model_GluGlc_after_FS.npy")

# In[6]: Updated schedule saving and plotting

Updated_schedule = {
    "Nacq": N_current_features,
    "B1_list": B1_list[remaining_features],
    "tsat_list": tsat_list[remaining_features],
    "Offsets_list": Offsets_list[remaining_features],
}

np.save("outputs/Updated_schedule.npy", Updated_schedule)

# Visualization of updated acquisition schedule
from Simulation_tools.Plotting_utils import plot_acquisition_schedule

plot_acquisition_schedule(
    B1_list[remaining_features],
    tsat_list[remaining_features],
    Offsets_list[remaining_features],
    saving=True,
    savename="outputs/New_acq_schedule.svg",
)
