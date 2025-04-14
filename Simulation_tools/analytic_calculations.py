# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:15:41 2022
@author: Cecile Maguin 

Subfunctions for analytic simulation
according to Bachert, Zaiss et al. 2013
"Exchange-dependent relaxation in the rotating frame for slow and intermediate exchange – modeling off-resonant spin-lock and chemical exchange saturation transfer"
DOI : 10.1002/nbm.2887
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import quad

from Simulation_tools.utils import gamma_2pi


class S_struc:  # for convenience, a structure to store apparent relaxation rates and flip angle
    def __init__(self):
        self.R1rho = np.empty()
        self.R1obs = np.empty()
        self.theta = np.empty()
        self.Rex = np.empty()


def calc_R1rho(sim):
    """ Main function to calculate global Relaxation rate R1rho of the full CEST system"""
    noffsets = np.size(sim.xZspec)

    S = S_struc

    S.Reff_sincos = -sim.CALC.R1_water * np.square(
        np.cos(sim.CALC.theta)
    ) - sim.CALC.R2_water * np.square(np.sin(sim.CALC.theta))

    if sim.shape == "block":  # block pulse solutions

        S.Rex = calc_full_Rex(sim)
        if sim.MT:
            [S.Rex_MT, S.R1obs] = calc_Rex_MT(sim, S.Reff_sincos, sim.CALC.w1)
            S.R1rho = S.Reff_sincos + S.Rex / (1 + sim.CALC.fH_MT) + S.Rex_MT
        else:
            S.R1rho = S.Reff_sincos + S.Rex
    else:  # other shapes of pulse

        integration_seg = 200  # number of integration points for the pulse shape
        B1 = RF_pulse(
            sim, integration_seg
        )  # decompose B1 all along the shape of the pulse

        # Calculate integrated Reff and Rex
        Reff_mean = np.zeros_like(sim.xZspec)
        Rex_mean = np.zeros(sim.n_cest_pools, noffsets)
        for i in range(noffsets):
            Reff_mean[i] = (
                np.trapz(
                    Reff_gauss(
                        sim.CALC.da[i],
                        gamma_2pi * B1,
                        sim.CALC.R2_water,
                        sim.CALC.R1_water,
                    )
                )
                / integration_seg
            )
            for j in range(sim.n_cest_pools):
                Rex_mean[j, i] = (
                    np.trapz(
                        Rex_Hyper_onepool(
                            sim.CALC.da[i],
                            gamma_2pi * B1,
                            sim.CALC.d_pools[j, i],
                            sim.kex[j],
                            sim.CALC.kbex[j],
                            sim.CALC.R2[j],
                        )
                    )
                    / integration_seg
                )
        # Calculate full Rex : sum over all pools
        S.Rex = np.sum(Rex_mean, 0)

        # Calculate R1rho with potential MT contribution
        if sim.MT == 1:
            S.R1obs = 0.5 * (
                sim.kex_MT
                + sim.CALC.kbex_MT
                + sim.CALC.R1_water
                + sim.CALC.R1_MT
                - np.sqrt(
                    np.square(
                        sim.CALC.kbex_MT
                        + sim.CALC.kex_MT
                        + sim.CALC.R1_water
                        + sim.CALC.R1_MT
                    )
                    - 4
                    * (
                        sim.CALC.kex_MT * sim.CALC.R1_water
                        + sim.CALC.kbex_MT * sim.CALC.R1_MT
                        + sim.CALC.R1_MT * sim.CALC.R1_water
                    )
                )
            )

            Rex_MT_full_B1 = np.zeros(integration_seg, noffsets)
            for j in range(integration_seg):
                Rex_MT_full_B1[j, :] = calc_Rex_MT(
                    sim,
                    Reff_gauss(
                        sim.CALC.da,
                        gamma_2pi * B1[j],
                        sim.CALC.R2_water,
                        sim.CALC.R1_water,
                    ),
                    gamma_2pi * B1[j],
                )
            Rex_mean_MT = np.trapz(Rex_MT_full_B1, axis=0) / integration_seg
            S.Rex_MT = Rex_mean_MT

            S.R1rho = S.Reff_sincos + S.Rex / (1 + sim.CALC.fH_MT) + S.Rex_MT
        else:  # No MT
            S.R1rho = Reff_mean + S.Rex
    return S


def calc_full_Rex(sim):
    """ Function to calculate CEST pools Rex relaxation rate """
    # Calculate relaxation rate Rex of all CEST pools

    if sim.n_cest_pools == 0:  # if no pools specified return 0
        return 0
    Individual_Rex = np.zeros((sim.n_cest_pools, np.size(sim.xZspec)))
    for j in range(sim.n_cest_pools):

        if sim.Rex_sol == "Lorentz":  # Lorentz solution
            Individual_Rex[j] = Rex_lorentz_onepool(
                sim.CALC.da,
                sim.CALC.w1,
                sim.CALC.d_pools[j, :],
                sim.kex[j],
                sim.CALC.kbex[j],
                sim.CALC.R2[j],
            )
        elif sim.Rex_sol == "Hyper":  # HyperLorentz solution
            Individual_Rex[j] = Rex_Hyper_onepool(
                sim.CALC.da,
                sim.CALC.w1,
                sim.CALC.d_pools[j, :],
                sim.kex[j],
                sim.CALC.kbex[j],
                sim.CALC.R2[j],
            )
        elif sim.Rex_sol == "minilorentz":  # Mini-Lorentz solution
            Individual_Rex[j] = Rex_minilorentz_onepool(
                sim.CALC.da,
                sim.CALC.w1,
                sim.CALC.d_pools[j, :],
                sim.kex[j],
                sim.CALC.kbex[j],
                sim.CALC.R2[j],
            )
        else:
            raise ValueError("Unrecognised Rex solution type")
        Rex = np.sum(Individual_Rex, 0)
    return Rex


def Rex_lorentz_onepool(da, w1, di, ki, ka, r2i):
    # calculate Rex Lorentz CEST solution for pool i
    # NBM paper;    assumes (kAB<<kBA, R1B<<kBA, Reff<<R2B)
    REXMAX = -((ka * (w1 ** 2)) / (np.square(da) + (w1 ** 2))) * (
        np.square(da - di) + (np.square(da) + (w1 ** 2)) * r2i / ki + r2i * (ki + r2i)
    )
    GAMMA = 2 * np.sqrt((1 + r2i / ki) * (w1 ** 2) + (ki + r2i) ** 2)
    Rex_Lorentz = REXMAX / (np.square(GAMMA / 2) + np.square(di))
    return Rex_Lorentz


def Rex_Hyper_onepool(da, w1, di, ki, ka, r2i):
    # calculate Rex HyperCEST solution for pool i
    # JCP paper;   assumes (R1B<<kBA, Reff<<R2B)
    Rex_hyper = -(
        ka
        * ki
        * (w1 ** 2)
        * (
            np.square(-da + di)
            + (r2i * (np.square(da) + (ka + ki) ** 2 + ki * r2i + (w1 ** 2))) / ki
        )
    ) / (
        (ka + ki) * (di ** 2 * (w1 ** 2) + ka * r2i * (w1 ** 2))
        + (ka + ki)
        * (
            (da * di - ka * r2i) ** 2
            + (di * ka + da * (ki + r2i)) ** 2
            + (ka + ki + r2i) ** 2 * (w1 ** 2)
        )
        + (ka + ki + r2i) * (np.square(da) * (w1 ** 2) + (w1 ** 4))
    )
    return Rex_hyper


def Rex_minilorentz_onepool(da, w1, di, ki, ka, r2i):
    # calculate Rex miniLorentz CEST solution for pool i
    REXMAX = -((ka * (w1 ** 2)) / (ki * (ki + r2i) + (w1 ** 2)))
    GAMMA = 2 * np.sqrt((1 + r2i / ki) * (w1 ** 2) + (ki + r2i) ** 2)
    Rex_minilorentz = REXMAX * (GAMMA / 2) ** 2 / ((GAMMA / 2) ** 2 + di ** 2)
    return Rex_minilorentz


def calc_Rex_MT(sim, Reff, w1):
    """ Functions to calculate MT Rex relaxation rate"""
    # To simplify variable calling
    da = sim.CALC.da
    dc = sim.CALC.dMT
    kCA = sim.kex_MT
    kAC = sim.CALC.kbex_MT
    r1a = sim.CALC.R1_water + Reff
    r2a = sim.CALC.R2_water + Reff
    r1c = sim.CALC.R1_MT + Reff

    # Find if the cutoff for MT lineshape is defined or not
    if hasattr(sim, "MT_cutoff"):
        MT_cutoff = sim.MT_cutoff
    else:
        MT_cutoff = (
            sim.CALC.w_ref
        )  # default MT cutoff if around +-1ppm of the MT resonance
    rfmt = RF_MT(sim.T2_MT, w1, dc, sim.MT_lineshape, MT_cutoff)
    # See Zaiss et al 2014
    Rex_MT = -(
        (
            (da ** 2 + r2a ** 2) * (kCA * r1a + (kAC + r1a) * (r1c + rfmt))
            + r2a * (kCA + r1c + rfmt) * w1 ** 2
        )
        / (
            da ** 2 * (kAC + kCA + r1a + r1c + rfmt)
            + r2a
            * (
                kCA * (2 * r1a + r2a)
                + r2a * (r1c + rfmt)
                + kAC * (2 * r1c + r2a + 2 * rfmt)
                + r1a * (2 * r1c + r2a + 2 * rfmt)
            )
            + (kCA + r1c + r2a + rfmt) * w1 ** 2
        )
    )

    R1obs = 0.5 * (
        kAC
        + kCA
        + sim.CALC.R1_water
        + sim.CALC.R1_MT
        - np.sqrt(
            (kAC + kCA + sim.CALC.R1_water + sim.CALC.R1_MT) ** 2
            - 4
            * (
                kCA * sim.CALC.R1_water
                + kAC * sim.CALC.R1_MT
                + sim.CALC.R1_water * sim.CALC.R1_MT
            )
        )
    )

    return Rex_MT, R1obs


def RF_pulse(sim, integration_seg):
    """ Functions for calculations in case of integrated pulse shape (i.e. non-block pulse shape)"""
    # Duty cycle
    DC = sim.tp / (sim.tp + sim.td)
    tpulse = np.linspace(0, sim.tp, integration_seg)

    if sim.shape == "gauss":
        t0 = sim.tp / 2
        w1max = sim.tp * sim.B1 / DC
        sig = sim.tp / 6
        w1 = (
            1
            / np.sqrt(2 * np.pi * sig ** 2)
            * w1max
            * np.exp(-((tpulse - t0) ** 2) / (2 * sig ** 2))
        )
    elif sim.shape == "Neurospin_seq":
        print("Need to redefine this sequence in the python code...")
    return w1


def Reff_gauss(da, w1, r2a, r1a):
    # Calculates Reff for pulsed gaussian solution
    Reff = -(
        r1a * da ** 2 / (da ** 2 + w1 ** 2) + (r2a) * w1 ** 2.0 / (da ** 2 + w1 ** 2)
    )
    return Reff


def RF_MT(T2c, w1, dw, lineshape, cutoff):
    """Calculates MT lineshape"""
    if isinstance(w1, float):
        w1 = [w1]
    rfmt = np.zeros((np.size(w1), np.size(dw)))
    if lineshape == "SuperLorentzian":
        for i in range(len(w1)):
            rfmt[i, :] = superlorentzian_shape(T2c, w1[i], dw, cutoff)
    elif lineshape == "Gaussian":
        rfmt = w1 ** 2 * T2c * np.sqrt(np.pi / 2) * np.exp(-((dw * T2c) ** 2) / 2)
    elif lineshape == "Lorentzian":
        rfmt = w1 ** 2 * T2c / (1 + (dw * T2c) ** 2)
    else:
        print("Unknown MT-lineshape - choose SuperLorentzian, Lorentzian or Gaussian")
        return 0
    return rfmt


def superlorentzian_shape(t2b, w1, delta, cutoff):
    if np.size(delta) == 1:
        rfmt = superlorentzian(t2b, w1, delta, cutoff)
    else:
        rfmt = np.zeros_like(delta)
        for j in range(np.size(delta)):
            rfmt[j] = superlorentzian(t2b, w1, delta[j], cutoff)
    return rfmt


def superlorentzian(t2b, w1, dw, cutoff):
    """Calculates Superlorentzian MT lineshape and includes cutoff around resonance value"""
    if abs(dw) >= cutoff:
        # see Morrsion and Henkelman 1995.  Need to multiply by w1^2 * pi to get saturation rate.
        # X=np.linspace(0,1,500) #200 integration points
        # Y=np.sqrt(2/np.pi)*t2b/np.abs(3*(X**2)-1) * np.exp(-2*(dw*t2b/(3*(X**2)-1))**2)
        def integrand(X):
            return (
                np.sqrt(2 / np.pi)
                * t2b
                / np.abs(3 * (X ** 2) - 1)
                * np.exp(-2 * (dw * t2b / (3 * (X ** 2) - 1)) ** 2)
            )

        # SL=w1**2 *np.pi *np.trapz(Y,X)
        integral = quad(integrand, 0, 1)
        SL = w1 ** 2 * np.pi * integral[0]
    else:  # if inside the cutoff zone, the shape is interpolated. This avoids infinite pole
        Xnew = [-1.1 * cutoff, -cutoff, cutoff, 1.1 * cutoff]
        Y = superlorentzian_shape(t2b, w1, Xnew, cutoff)
        SL_intfct = interpolate.interp1d(Xnew, Y, "quadratic")
        SL = SL_intfct(dw)
    return SL
