# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:08:13 2022

@author: Cecile Maguin 

Analytic calculation of CEST solution, according to Bachert, Zaiss et al. 2013
"Exchange-dependent relaxation in the rotating frame for slow and intermediate exchange – modeling off-resonant spin-lock and chemical exchange saturation transfer"
DOI : 10.1002/nbm.2887
"""
import numpy as np
from copy import deepcopy

from Simulation_tools.utils import *
from Simulation_tools.analytic_calculations import *


def analytic_simulation(P, override=0):

    """This function computes the CEST simulation with all the parameters indicated in the P simulation structure -
    Returns simulated data values in Zspec and M0 value computed from P.normalized"""

    sim = deepcopy(P)

    """Preparatory checks"""
    # B1 correction if necessary
    # CAREFUL : this must be done only once in the code
    if hasattr(sim, "B1c"):
        sim.B1 = sim.B1 * sim.B1c
    # adjust resonance frequencies from water resonance
    # CAREFUL : this must be done only once in the code
    sim.dw = sim.dw + sim.dw_water

    # Check that the sim structure is well updated
    sim.update()

    M0 = 1
    # Experimental normalization
    if hasattr(sim, "normalized"):
        simM0 = deepcopy(sim)
        simM0.xZspec = [sim.normalized]
        M0 = analytic_solution(simM0)
    """If full relaxation between offsets approximation is not assumed, then sequential computing with Zi adjustment """
    if hasattr(sim, "Offsets_order"):
        if sim.Offsets_order == "successive":
            xZspec = sim.xZspec
            Zspec = np.zeros_like(xZspec)
            simloc = deepcopy(sim)
            i = 0
            simloc.Zi = M0  # we consider M0 is measured first
            for offset in xZspec:
                simloc.xZspec = [offset]
                Zoff = analytic_solution(simloc)
                Zspec[i] = Zoff
                i += 1
                simloc.Zi = Zoff
        else:
            print("Did not programm the other possibilities of offsets order. ")
            return -1
    else:
        Zspec = analytic_solution(sim)
    if hasattr(
        sim, "WaterResCorr"
    ):  # correct full water saturation at water resonance (when needed)
        if sim.WaterResCorr == 1:
            # xZspec_to_correct=np.extract(np.abs(xZspec)<0.25, xZspec)
            for i in range(np.size(sim.xZspec)):
                if np.size(sim.xZspec) > 1:
                    offset = sim.xZspec[i]
                else:
                    offset = sim.xZspec
                if np.abs(offset) < 0.25:
                    simlocw = deepcopy(sim)
                    xZspec_for_interp = np.concatenate(
                        (
                            np.arange(-1.5, -0.2, 0.05),
                            np.array([sim.dw_water]),
                            np.arange(0.25, 1.55, 0.1),
                        )
                    )
                    simlocw.xZspec = xZspec_for_interp
                    Z_for_interp = analytic_solution(simlocw)
                    Z_for_interp[26] = 0

                    Zspec[i] = np.interp(offset, xZspec_for_interp, Z_for_interp)
    # add normalization at the end
    Zspec = Zspec / M0

    return Zspec, M0


def analytic_solution(sim):

    """Calculates the explicit analytic solution of CEST problem - Returns solution in Zspec"""

    sim.update()

    # Correct initial magnetisation
    Zi = (sim.Zi - 1) * np.exp(-sim.CALC.R1_water * (sim.Trec - sim.td)) + 1

    """Calculate solution"""
    # Calculate global relaxation rate
    S = calc_R1rho(sim)
    R1rho = S.R1rho
    theta = sim.CALC.theta

    # depending on the shape of the relaxation module, assign Pz and Pzeff
    if sim.shape == "SL":  # spinlocking case
        Pzeff = 1
        Pz = 1
    else:
        Pzeff = np.cos(theta)
        Pz = np.cos(theta)
    # If MT is included in the simulation, add correction to R1 according to Zaiss 2015 doi:10.1002/nbm.3237
    if sim.MT == 1:
        R1obs = S.R1obs
    else:
        R1obs = sim.CALC.R1_water
    # Continuous wave irradiation case
    if sim.pulsed == 0:
        Zspec = (Zi + np.divide(Pz * Pzeff * R1obs, R1rho)) * np.exp(
            R1rho * sim.tp
        ) - np.divide(Pz * Pzeff * R1obs, R1rho)
    # Pulsed sequence solution : see Santyr 1994 and Zaiss/Bachert 2012
    else:
        # Zss = 1-  (1-np.exp(R1rho.flatten()*sim.tp)) * (1-np.cos(theta)) *sim.CALC.R1_water /( -R1rho.flatten()*( 1-np.exp(R1rho.flatten()*sim.tp-sim.CALC.R1_water*sim.td)))
        Zss = 1 - (
            (1 - np.exp(R1rho * sim.tp))
            * (1 - np.cos(theta) * sim.CALC.R1_water / (-R1rho))
        ) / (1 - np.exp(R1rho * sim.tp - sim.CALC.R1_water * sim.td))
        Zspec = Zss + (Zi - Zss) * np.exp(
            sim.n * (R1rho * sim.tp - sim.CALC.R1_water * sim.td)
        )
    """Additionnal effects that can be taken into account"""
    # Relaxation before readout
    if sim.play_readout == 1:
        if hasattr(sim, "readout_delay"):
            Zspec = 1 + (Zspec - 1) * np.exp(-sim.readout_delay / sim.T1_water)
        if hasattr(sim, "TEeff") == "False":
            sim.TEeff = 0.03  # default TE in our sequence
        Zspec = 1 + (Zspec - 1) * np.exp(-sim.TEeff / sim.T1_water)
    return Zspec


# def analytic_simu_successive_offsets(sim):
#     """ This can be used to compute the solution when incomplete relaxation is reached between two offsets"""

#     xZspec=sim.xZspec
#     Zspec=np.zeros_like(xZspec)
#     i=0
#     #we consider M0 is measured first
#     sim.xZspec=[sim.normalized]
#     M0, dump=analytic_simulation(sim, 1)
#     for offset in xZspec:
#         sim.xZspec=[offset]
#         Zoff,dump=analytic_simulation(sim, 1)
#         Zspec[i]=Zoff
#         i+=1
#         sim.Zi=Zoff
#     Zspec=Zspec/M0

#     return Zspec,M0
