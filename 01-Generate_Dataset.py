# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:08:33 2022
@author: Cecile Maguin 

Let's try to create a realistic and representative data set for NN training for quantitative CEST reconstruction.
We first define the basics of the simulations; the acquisition parametrization; then specify the kind of scenario we want to generate.
Then we generate simulations and save them in dataset for later. 
"""

# In[1]: Imports

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import date
import time
from tqdm import tqdm


from Simulation_tools.utils import simulation, gamma


# Choose random seed
current_seed = 136
random.seed(current_seed)

# In[2]: Define Simulation basics


# Init simulation structure and global parameters
Sim = simulation()
Sim.FREQ = 11.75 * gamma  # frequence of MRI scanner (here 11.7T)
Sim.Zi = 1  # Initial magnetization (beetween -1 and +1)
Sim.B1 = 3  # B1 in µT
Sim.pulsed = 1  # pulsed (1) or continuous (0) saturation
Sim.shape = "block"  # shape of saturation pulse (only 'block' or 'gauss' coded for now)
Sim.n = 10  # number of pulses (if CW, then n=1)
Sim.tp = 0.1  # Duration of each saturation pulse in seconds
Sim.td = 1e-5  # Inter-pulse delay (if CW, then td=0)
Sim.Trec = 4  # Recovery time (useful only if you consider that full relaxation is not achieved between two offsets mesurements)
# Sim.normalized=-8   #Offset in ppm of the M0 used for normalization (if not given, then we consider perfect M0=1)

# Choose CEST models
Sim.add_pool_model("Glucose_phantom")
Sim.add_pool_model("Glutamate")
Sim.WaterResCorr = 1  # 1=ON / 0=OFF - this to correct for numerical instabilities around water resonance (simple interpolation)

Sim.T1_water = 2.0  # Unit=seconds
Sim.T2_water = 0.8  # Unit=seconds

Sim.kex_Glc_1 = 500  # Unit=Hz
Sim.kex_Glc_2 = 450
Sim.kex_Glc_3 = 450
Sim.kex_Glc_4 = 600

# In[3]: Acquisition Parametrization

# Choose acquistion points coordinates
xZspec = np.arange(-5, 5.1, 0.1)
# xZspec=np.append(xZspec,-20) #if you add the M0 measurement too

B1s = np.array([3, 5, 7, 3, 5, 4, 6])
B1_list = np.repeat(B1s, len(xZspec))

tsats = np.array([1, 1, 1, 4, 4, 1, 1])
tsat_list = np.repeat(tsats, len(xZspec))

Offsets_list = np.tile(xZspec, len(B1s))

Nacq = len(B1_list)

print("Scheduled B1 : \n")
print(B1_list)
print("Scheduled tsat : \n")
print(tsat_list)
print("Scheduled Offsets : \n")
print(np.round(Offsets_list, 1))

# Store this parametrization in dict for convenience
Acq_Param = {
    "Nacq": Nacq,
    "B1_list": B1_list,
    "tsat_list": tsat_list,
    "Offsets_list": Offsets_list,
}

# Visualization of acquisition schedule
from Simulation_tools.Plotting_utils import plot_acquisition_schedule

plot_acquisition_schedule(
    B1_list,
    tsat_list,
    Offsets_list,
    saving=True,
    savename="outputs/Initial_acq_schedule.svg",
)

# In[4]: Dataset specifications

# Number of data scenarii to generate
N_data = 5000

# Variables lower and upper range
Var_range = {
    "Glu": {"min": 0, "max": 60},
    "kex_Glu": {"min": 4000, "max": 5000},
    "Glc": {"min": 0, "max": 60},
    "kex_Glc_1": {"min": 200, "max": 1000},
    "kex_Glc_2": {"min": 200, "max": 1000},
    "kex_Glc_3": {"min": 200, "max": 1000},
    "kex_Glc_4": {"min": 200, "max": 1000},
    "T1_water": {"min": 1.8, "max": 2.5},
    "T2_water": {"min": 0.2, "max": 1.0},
    "dw_water": {"min": -0.2, "max": 0.2},
    "B1_noise": {"min": 0.98, "max": 1.02},
    "dw_water_drift": {"min": -0.02, "max": 0.02},
}

Values_of_interest = ["Glu", "Glc"]  # variables we want to quantify
Ratio_fact = [1, 1]  # optional weighting


# In[5]: Dictionnary definition

from NN_quantif_tools.Data_generation_utils import create_biological_scenario

Ground_truth = np.zeros((N_data, np.size(Values_of_interest)))

S_list = np.zeros((N_data, 1), dtype=dict)

# First create dictionnary of biological scenarii
for n in range(N_data):
    Scenar = create_biological_scenario(
        Var_range, rng_seed=random.randint(0, 1000000), std_noise=0.001, Nacq=Nacq
    )

    # Extract ground truth of variables of interest
    for ii in range(len(Values_of_interest)):
        Ground_truth[n, ii] = Ratio_fact[ii] * Scenar[Values_of_interest[ii]]
    S_list[n] = Scenar
# In[6]: Generate simulation for every entry in the dictionnary of scenarii

from NN_quantif_tools.Data_generation_utils import generate_simulations

Data = np.zeros((N_data, Nacq))

tic = time.perf_counter()

# # If small data set, can just simulate it directly
# for n in tqdm(range(N_data)):
#     Data[n, :] = generate_simulations(S_list[n], Sim, Acq_Param)

# Parallelized version
from joblib import Parallel, delayed

n_k = 12  # -1 for all cores
Data = Parallel(n_jobs=n_k)(
    delayed(generate_simulations)(S_list[n], Sim, Acq_Param)
    for n in tqdm(range(N_data))
)
Data = np.reshape(Data, (N_data, Nacq))

toc = time.perf_counter()
print("100% of data set is generated (Total computing time :" + str(toc - tic) + "s)")


# In[7]: Plot a few examples of generated data
plt.figure()
graph, plot1 = plt.subplots()
xaxis = np.arange(1, Nacq + 1)
list_plot = np.random.randint(0, N_data, 3)  # random examples
for iplot in list_plot:
    plot1.plot(
        xaxis,
        Data[iplot, :],
        label="[Glc]="
        + str(round(Ground_truth[iplot, 0], 1))
        + " mM, [Glu]="
        + str(round(Ground_truth[iplot, 1], 1))
        + " mM, etc..",
    )
plot1.yaxis.label.set_fontsize(12)
plot1.set_ylabel("M$_z$/M$_0$")
plot1.legend(loc="lower left", fontsize="large")
plot1.xaxis.label.set_fontsize(12)
plot1.tick_params(labelsize=18)
plt.show()

# In[8]: Save

import os

savepath = "Data/"
# Choose name
Name = savepath + "Dataset_GluGlc_" + str(date.today()) + "/"
flag = 0
if os.path.exists(Name):
    ans = input("Probably overwriting previous dataset - ok with that ? (Y) \n")
    if ans == "Y":
        flag = 1
else:
    flag = 1
    os.mkdir(Name)
# Proceed with saving
if flag == 1:
    # Save Data set
    np.save(Name + "Data", Data)
    np.save(Name + "Ground_truth_for_training", Ground_truth)

    np.save(Name + "Detailed_biological_scenarii", S_list, allow_pickle=True)

    # Save simulation template
    np.save(Name + "Simulation_template", Sim, allow_pickle=True)
    np.save(Name + "Acq_param", Acq_Param)

    # Save info about dataset
    textfile = open(Name + "Dataset_info.txt", "w")
    textfile.write("Dataset generated on " + str(date.today()) + "\n")
    textfile.write("Dataset includes " + str(N_data) + " examples \n")
    textfile.write("Random seed is " + str(current_seed) + "\n")
    textfile.write("B1 list (µT) : \n" + "".join(str(B1_list).splitlines()) + "\n")
    textfile.write("tsat list (s) : \n" + "".join(str(tsat_list).splitlines()) + "\n")
    textfile.write(
        "Offsets list (ppm) : \n" + "".join(str(Offsets_list).splitlines()) + "\n"
    )
    textfile.close()
