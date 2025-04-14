# -*- coding: utf-8 -*-
"""
Dealing with dataset generation

Created on Wed Jul 20 10:31:28 2022

@author: Cecile Maguin
"""
import numpy as np
import numpy.random as rng

from Simulation_tools.analytic_sim import analytic_simulation


def create_biological_scenario(
    Var_range,
    rng_seed=rng.randint(0, 1000000),
    std_noise=0.01,
    Nacq=30,
    Drift_periodicity=1,
):
    """Generate a biological and experimental scenario with given variables 
    that can vary between Var_range["min] and Var_range["max"]. 
    Additionnally some experimental noise is defined with typical std std_noise.
    Drift periodicity corresponds to the periodicity when water drift correction 
    is performed experimentally (for instance, =30 if it is corrected every 30 offsets)
    """

    rng.seed(rng_seed)

    Scenario = dict()

    Var_list = list(Var_range.keys())
    N_var = len(Var_list)

    Scenario["seed"] = rng_seed

    # Randomly choose variables for this scenario in given range (uniform law - should I change to gaussian ?)
    for v in Var_range.keys():
        if v == "B1_noise":
            Scenario[v] = (
                rng.random(Nacq) * (Var_range[v]["max"] - Var_range[v]["min"])
                + Var_range[v]["min"]
            )
        elif v == "dw_water_drift":
            drift = np.zeros(Nacq)
            for dr in range(int(np.ceil(Nacq / Drift_periodicity))):
                drift[
                    dr * Drift_periodicity : min(Nacq, (dr + 1) * Drift_periodicity)
                ] = (
                    rng.random() * (Var_range[v]["max"] - Var_range[v]["min"])
                    + Var_range[v]["min"]
                ) * np.ones(
                    min(Drift_periodicity, Nacq - dr * Drift_periodicity)
                )
            Scenario[v] = drift
        else:
            Scenario[v] = (
                rng.random() * (Var_range[v]["max"] - Var_range[v]["min"])
                + Var_range[v]["min"]
            )
        # Predefine the noise during acquisition (Gaussian noise for CEST works well)
        Scenario["acq_noise"] = rng.normal(1.0, std_noise, Nacq)
    return Scenario


def generate_simulations(S_list, Sim, Acq_Param):
    """Simulate Scenarios according to config indicated in Sim and for acquisition points specified in Acq_param"""
    N_data = np.size(S_list)

    Current_sim_config = Sim

    Sim_data = np.zeros((N_data, Acq_Param["Nacq"]))

    for n in range(N_data):
        # Current scenario
        scenar = S_list[n]
        # set up variable values of the biological scenario
        for v in scenar.keys():
            setattr(Current_sim_config, v, scenar[v])
        Current_sim_config.update()

        # now simulate for each step of the schedule the value of acquisition
        for acq in range(Acq_Param["Nacq"]):
            # change schedule values
            Current_sim_config.B1 = Acq_Param["B1_list"][acq]
            Current_sim_config.n = np.round(
                Acq_Param["tsat_list"][acq] / Current_sim_config.tp
            )  # for our setup each pulse is 0.1s
            Current_sim_config.xZspec = Acq_Param["Offsets_list"][acq]

            # add some noise on B1 if specified
            if hasattr(Current_sim_config, "B1_noise"):
                Current_sim_config.B1 = (
                    Current_sim_config.B1 * Current_sim_config.B1_noise[acq]
                )
            if hasattr(Current_sim_config, "dw_water_drift"):
                if acq == 0:
                    original_value = Current_sim_config.dw_water
                Current_sim_config.dw_water = (
                    original_value + Current_sim_config.dw_water_drift[acq]
                )
            Acq_point, M0 = analytic_simulation(Current_sim_config)
            Acq_point = Acq_point * scenar["acq_noise"][acq]

            # if global M0 noise is pecified, modifiy output
            if hasattr(Current_sim_config, "M0_global"):
                Acq_point = Acq_point * Current_sim_config.M0_global
            Sim_data[n, acq] = Acq_point
    return Sim_data


def read_info(info_path):
    """A function to re-extract acquisition schedule from info textfile of a dataset"""

    # Load info and relevant variables
    textfile = open(info_path, "r")
    infos = textfile.readlines()

    # Ndata = int(infos[1][17:-11])

    L = infos[4].strip()
    B1_list = np.array(L[1:-1].split(), dtype="float64")
    L = infos[6].strip()
    tsat_list = np.array(L[1:-1].split(), dtype="float64")
    L = infos[8].strip()
    Offsets_list = np.array(L[1:-1].split(), dtype="float64")

    textfile.close()

    Nacq = len(B1_list)

    return Nacq, B1_list, tsat_list, Offsets_list


# For advanced testing of optimization (autoCEST paper)
# class data_generation_layer(th.nn.Module):
#     def __init__(
#         self, Nacq, sim, parameters=["B1"], schedule_init=["random"]
#     ):  # possible_parameters=['B1','tsat','Offset']

#         super(data_generation_layer, self).__init__()

#         self.Nacq = Nacq
#         self.sim = sim

#         # Variable schedule parameters, registered as torch parameters to be optimized
#         self.acq_parameters = parameters

#         for i in range(np.size(self.acq_parameters)):
#             # print(self.acq_parameters[i])
#             if self.acq_parameters[i] == "B1":
#                 if isinstance(schedule_init[i], str):
#                     if schedule_init[i] == "random":
#                         self.B1_list = th.nn.Parameter(
#                             th.rand(Nacq) * 10
#                         )  # B1 schedule randomly chosen between 0 and 10 µT by default
#                 else:
#                     self.B1_list = th.nn.Parameter(th.FloatTensor(schedule_init[i]))
#                 # record initial schedule for comparison purposes
#                 self.initial_B1_list = th.clone(self.B1_list)
#             elif self.acq_parameters[i] == "tsat":
#                 if isinstance(schedule_init[i], str):
#                     if schedule_init[i] == "random":
#                         self.tsat_list = th.nn.Parameter(
#                             th.randint(0, 50, (Nacq,)) / 10
#                         )  # tsat schedule randomly chosen between 0.1 and 5 s by steps of 0.1s by default
#                 else:
#                     self.tsat_list = th.nn.Parameter(th.FloatTensor(schedule_init[i]))
#                 self.initial_tsat_list = th.clone(self.tsat_list)
#             elif self.acq_parameters[i] == "Offset":
#                 if isinstance(schedule_init[i], str):
#                     if schedule_init[i] == "random":
#                         self.Offset_list = th.nn.Parameter(
#                             th.rand(Nacq) * 10 - 5 * th.ones(Nacq)
#                         )  # f schedule randomly chosen between -5 and 5 ppm by default
#                 else:
#                     self.Offset_list = th.nn.Parameter(th.FloatTensor(schedule_init[i]))
#                 self.initial_Offset_list = th.clone(self.Offset_list)

#     def forward(self, biological_scenarii):

#         N_data = np.size(biological_scenarii)

#         Data = np.zeros((N_data, self.Nacq))

#         for n in range(N_data):
#             if N_data == 1:
#                 scenario = biological_scenarii[0]
#             else:
#                 scenario = biological_scenarii[n, 0]
#             # set up variable values of the biological scenario
#             for attribute in scenario.variables:
#                 setattr(self.sim, attribute, getattr(scenario, attribute))
#             self.sim.update()

#             # now simulate for each step of the schedule the value of acquisition
#             for acq in range(self.Nacq):

#                 # change schedule values
#                 for param in self.acq_parameters:
#                     if param == "B1":
#                         self.sim.B1 = self.B1_list[acq].detach().numpy()
#                     if param == "tsat":
#                         self.sim.n = np.round(
#                             self.tsat_list[acq].detach().numpy() / self.sim.tp
#                         )  # for our setup each pulse is 0.1s
#                     if param == "Offset":
#                         self.sim.xZspec = self.Offset_list[acq].detach().numpy()
#                 # print(self.sim.B1)
#                 # print(self.sim.n)
#                 # print(self.sim.xZspec)
#                 # add some noise on B1 if specified
#                 if hasattr(self.sim, "B1_noise"):
#                     self.sim.B1 = self.sim.B1 * self.sim.B1_noise[acq]
#                 if hasattr(self.sim, "dw_water_drift"):
#                     if acq == 0:
#                         original_value = self.sim.dw_water
#                     self.sim.dw_water = original_value + self.sim.dw_water_drift[acq]
#                 Acq_point, M0 = analytic_simulation(self.sim)

#                 Acq_point = Acq_point * scenario.acq_noise[acq]

#                 # if global M0 noise is pecified, modifiy output
#                 if hasattr(self.sim, "M0_global"):
#                     Acq_point = Acq_point * self.sim.M0_global
#                 Data[n, acq] = Acq_point
#         # print(th.FloatTensor(Data))
#         return th.FloatTensor(Data)
