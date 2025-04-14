# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:28:31 2022

@author: Cecile Maguin 

Utils for qCEST simulation pipeline
"""

import numpy as np
from numpy import pi

import pickle

# import traceback

global gamma, gamma_2pi
gamma = 42.576375
gamma_2pi = 2 * pi * gamma


class simulation:
    "A class which contains all information about the simulation parameters"

    def __init__(self, WaterConc=55556):

        # Simulation options
        self.analytic = 1  # by default the simulation is analytic - in this kind of problem it does not deviate too much from numeric simulation so it should be okay
        self.MT = 0  # by default there is no MT pool
        self.Rex_sol = "Lorentz"  # by default the simulation is calculated with Rex Lorentz solution
        self.play_readout = 0  # by default we assume a perfect instantaneous readout
        self.xZspec = np.array([])  # Offsets list

        # Basic parameters initialisation
        self.n_cest_pools = 0  # Number of CEST pools (water & MT excluded)
        self.pool_names = np.array([])
        self.H = np.array([])  # Concentrations in mM
        self.kex = np.array([])  # Exchange rates in Hz
        self.dw = np.array([])  # Resonance frequencies (ppm)
        self.T1 = np.array([])  # T1 in s
        self.T2 = np.array([])  # T2 in s

        # Add a water pool
        self.model = np.array(["Water"])
        self.dw_water = 0  # water resonance (ppm)
        self.H_water = (
            2 * WaterConc
        )  # water proton concentration (mM) - default is pure water
        self.T1_water = 2.2  # Water T1 (s)
        self.T2_water = 0.1  # Water T2 (s)

    def add_constraint(self, relation):
        """Can add new constraint to the problem (for instance, on relationship between pools, or on pH, temperature,...). 
        For flexibility, we store them as str constraints. 
        These constraints will be recalculated when sim.update is performed"""
        if hasattr(self, "constraints"):
            self.constraints = np.append(self.constraints, relation)
        else:
            self.constraints = np.array([relation])

    def add_pool_model(self, pool_name):
        """Add a new chemical species to the CEST problem. Some basic pools are available but you can add more. """

        self.model = np.append(self.model, pool_name)

        if pool_name == "MT":
            self.MT = 1
            self.MT_lineshape = "SuperLorentzian"
            self.H_MT = 0.05 * 2 * 55.6 * 1000  # by default 5% MT
            self.kex_MT = 40  # Hz
            self.dw_MT = -2.34  # ppm
            self.T1_MT = 1  # s
            self.T2_MT = 1e-5  # s
        elif pool_name == "Glutamate":
            if (
                hasattr(self, "Glu") == 0
            ):  # if concentration is not given, initialize [Glu] at 10mM
                self.Glu = 10  # Unit=mM
            self.pool_names = np.append(self.pool_names, "Glu")
            self.H = np.append(self.H, 3 * self.Glu)  # Unit=mM
            self.kex = np.append(self.kex, 7738)  # Unit=Hz
            self.dw = np.append(self.dw, 3.0)  # Unit=ppm
            self.T1 = np.append(self.T1, self.T1_water)  # Unit=s
            self.T2 = np.append(self.T2, self.T2_water)  # Unit=s

            self.add_constraint("self.H[" + str(self.n_cest_pools) + "]=3*self.Glu")

            self.n_cest_pools += 1
        elif (
            pool_name == "Glucose_phantom"
        ):  # as given in Zaiss 2019, at approx pH=7 and 25° (phantoms)
            if (
                hasattr(self, "Glc") == 0
            ):  # if concentration is not given, initialize [Glc] at 10mM
                self.Glc = 10  # Unit=mM
            if (
                hasattr(self, "AR") == 0
            ):  # anomeric ratio of alpha/beta forms of glucose
                self.AR = 0.36
            self.pool_names = np.append(
                self.pool_names, ["Glc_1", "Glc_2", "Glc_3", "Glc_4"]
            )

            self.H = np.append(
                self.H,
                [self.Glc, 3 * self.Glc, self.Glc * self.AR, self.Glc * (1 - self.AR)],
            )  # Unit=mM
            self.kex = np.append(self.kex, [2300, 5000, 3800, 10000])  # Unit=Hz
            self.dw = np.append(self.dw, [0.75, 1.29, 2.18, 2.88])  # Unit=ppm
            self.T1 = np.append(
                self.T1, [self.T1_water, self.T1_water, self.T1_water, self.T1_water]
            )  # Unit=s
            self.T2 = np.append(self.T2, [0.008, 0.015, 0.015, 0.015])  # Unit=s

            self.add_constraint("self.H[" + str(self.n_cest_pools) + "]=self.Glc")
            self.add_constraint("self.H[" + str(self.n_cest_pools + 1) + "]=3*self.Glc")
            self.add_constraint(
                "self.H[" + str(self.n_cest_pools + 2) + "]=self.AR*self.Glc"
            )
            self.add_constraint(
                "self.H[" + str(self.n_cest_pools + 3) + "]=(1-self.AR)*self.Glc"
            )

            self.n_cest_pools += 4
        elif (
            pool_name == "Glucose_physio"
        ):  # as given in Zaiss et al. 2019 Table 2 (pH 7.2 and 37°C)
            if (
                hasattr(self, "Glc") == 0
            ):  # if concentration is not given, initialize [Glc] at 10mM
                self.Glc = 3  # Unit=mM
            if (
                hasattr(self, "AR") == 0
            ):  # anomeric ratio of alpha/beta forms of glucose
                self.AR = 0.38
            self.pool_names = np.append(
                self.pool_names, ["Glc_1", "Glc_2", "Glc_3", "Glc_4"]
            )

            self.H = np.append(
                self.H,
                [self.Glc, 3 * self.Glc, self.Glc * self.AR, self.Glc * (1 - self.AR)],
            )  # Unit=mM
            self.kex = np.append(self.kex, [2900, 6500, 5200, 14000])  # Unit=Hz
            self.dw = np.append(self.dw, [0.74, 1.29, 2.18, 2.88])  # Unit=ppm
            self.T1 = np.append(
                self.T1, [self.T1_water, self.T1_water, self.T1_water, self.T1_water]
            )  # Unit=s
            self.T2 = np.append(self.T2, [0.008, 0.015, 0.015, 0.015])  # Unit=s

            self.add_constraint("self.H[" + str(self.n_cest_pools) + "]=self.Glc")
            self.add_constraint("self.H[" + str(self.n_cest_pools + 1) + "]=3*self.Glc")
            self.add_constraint(
                "self.H[" + str(self.n_cest_pools + 2) + "]=self.AR*self.Glc"
            )
            self.add_constraint(
                "self.H[" + str(self.n_cest_pools + 3) + "]=(1-self.AR)*self.Glc"
            )

            self.n_cest_pools += 4
        elif pool_name == "Creatine":
            if (
                hasattr(self, "Cr") == 0
            ):  # if concentration is not given, initialize [Glu] at 10mM
                self.Cr = 10  # Unit=mM
            self.pool_names = np.append(self.pool_names, "Cr")
            self.H = np.append(self.H, 4 * self.Cr)  # Unit=mM
            self.kex = np.append(self.kex, 950)  # Unit=Hz
            self.dw = np.append(self.dw, 2.0)  # Unit=ppm
            self.T1 = np.append(self.T1, self.T1_water)  # Unit=s
            self.T2 = np.append(self.T2, self.T2_water)  # Unit=s

            self.add_constraint("self.H[" + str(self.n_cest_pools) + "]=4*self.Cr")

            self.n_cest_pools += 1
        else:
            raise ValueError(
                "Unknown CEST pool - check spelling or input your own CEST pool"
            )

    """ ...... TO FILL if more models needed........."""

    def convert_names_to_variables(self):
        """This is to convert to real variables if there are some pseudo-code variables added"""

        for i in range(self.n_cest_pools):
            suffix = self.pool_names[i]

            if hasattr(self, "kex_" + suffix):
                self.kex[i] = eval("self.kex_" + suffix)
            if hasattr(self, "T2_" + suffix):
                self.T2[i] = eval("self.T2_" + suffix)
            if hasattr(self, "T1_" + suffix):
                self.T1[i] = eval("self.T1_" + suffix)
            if hasattr(self, "dw_" + suffix):
                self.dw[i] = eval("self.dw_" + suffix)
            if hasattr(self, "H_" + suffix):
                self.H[i] = eval("self.H_" + suffix)

    def update(self):
        """Update simulation and make sure everything is consistent"""

        # Convert pseudo code to variables
        self.convert_names_to_variables()

        # convert ot numpy arrays
        self.pool_names = np.array(self.pool_names)
        self.H = np.array(self.H)
        self.kex = np.array(self.kex)
        self.dw = np.array(self.dw)
        self.T1 = np.array(self.T1)
        self.T2 = np.array(self.T2)

        # Check that xZspec is still numpy array (should be but...)
        self.xZspec = np.array(self.xZspec)

        # Execute constraints on the problem (pH, fractions of enantiomers, temperature,...)
        if hasattr(self, "constraints"):
            for i in range(0, len(self.constraints)):
                exec(self.constraints[i])
        self.calc_alt_var()

    def calc_alt_var(self):
        """Calculate alternate variables for computation and store in CALC"""
        self.CALC = calc_struc(self)

    def save(self, name):
        """Save sim in name file"""
        file = open(name + ".txt", "wb")
        pickle.dump(self.__dict__, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, name):
        """try load simulation by the name of"""
        file = open(name + ".txt", "rb")
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)


class calc_struc:
    """Structure for storing explicit variables for computation - can auto udpate itself"""

    def __init__(self, sim):
        # Calculate equivalent frequencies
        self.w_ref = 2 * pi * sim.FREQ
        self.w1 = sim.B1 * gamma_2pi

        # Translate T1 and T2 to R1, R2
        self.R1_water = 1 / sim.T1_water
        self.R2_water = 1 / sim.T2_water
        self.R1 = 1 / sim.T1
        self.R2 = 1 / sim.T2
        if sim.MT == 1:
            self.R1_MT = 1 / sim.T1_MT
            self.R2_MT = 1 / sim.T2_MT
        # Calculate chemical shifts from resonance frequencies
        self.da = (sim.xZspec - sim.dw_water) * self.w_ref
        self.d_pools = np.zeros((sim.n_cest_pools, np.size(sim.xZspec)))
        for i in range(0, sim.n_cest_pools):
            self.d_pools[i, :] = (sim.xZspec - sim.dw[i]) * self.w_ref
        if sim.MT == 1:
            self.dMT = (sim.xZspec - sim.dw_MT) * self.w_ref
        # Calculate equivalent flip angle
        self.theta = list(np.zeros(np.size(self.da)))
        for i in range(np.size(self.da)):
            if isinstance(self.da, float):
                di = self.da
            else:
                di = self.da[i]
            if di != 0:
                self.theta[i] = np.arctan(self.w1 / di)
            else:
                self.theta[i] = np.pi / 2
        self.da = np.array(self.da)
        self.theta = list(self.theta)

        # Calculate proton fractions
        self.fH = sim.H / sim.H_water
        if sim.MT == 1:
            self.fH_MT = sim.H_MT / sim.H_water
        # Calculate backward exchange rates
        self.kbex = sim.kex * self.fH
        if sim.MT == 1:
            self.kbex_MT = sim.kex_MT * self.fH_MT

    def update_local(self):
        # Update equivalent flip angle
        self.theta = list(np.zeros(np.size(self.da)))
        for i in range(np.size(self.da)):
            if isinstance(self.da, float):
                di = self.da
            else:
                di = self.da[i]
            if di != 0:
                self.theta[i] = np.arctan(self.w1 / di)
            else:
                self.theta[i] = np.pi / 2
        self.da = np.array(self.da)
        self.theta = list(self.theta)
