# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:26:37 2022

@author: Cecile Maguin 

Utilities for plotting nice fig 
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_acquisition_schedule(
    B1_list, tsat_list, Offsets_list, saving=False, savename="outputs/Acq_schedule.svg"
):

    """ Generic plotting of data acquisition schedule - Can save image in svg if saving set to True"""

    Nacq = len(B1_list)

    plt.style.use("ggplot")
    plt.figure()
    plt.rcParams["figure.figsize"] = [8, 3]
    plt.rcParams["figure.autolayout"] = True

    fs = 22

    fig, ax1 = plt.subplots()
    ax1.yaxis.grid(False)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_visible(False)
    ax1.set_facecolor("0.93")
    xaxis = np.arange(1, Nacq + 1)

    ax11 = ax1.twinx()
    ax11.spines.right.set_position(("axes", 1.1))
    ax11.yaxis.grid(False)
    ax1.xaxis.label.set_fontsize(fs)
    # plt.xticks(xaxis)
    ax1.tick_params(labelsize=15)
    ax11.yaxis.set_visible(True)

    # B1 schedule
    # plot line chart on axis #1
    (p1,) = ax11.plot(xaxis, B1_list, "C0")
    ax11.set_ylabel("B$_1$ (µT)")
    plt.yticks([1, 2, 3, 4, 5, 6, 7])
    ax11.set_ylim(0, 10)
    # ax11.legend(['Saturation power'], loc="upper left",fontsize='x-large')
    ax11.yaxis.label.set_color(p1.get_color())
    ax11.yaxis.label.set_fontsize(fs)
    ax11.tick_params(axis="y", colors=p1.get_color(), labelsize=18)

    # set up the 2nd axis
    ax2 = ax1.twinx()
    # plot bar chart on axis #2
    (p2,) = ax2.plot(xaxis, tsat_list, "C1")
    ax2.grid(False)  # turn off grid #2
    ax2.set_ylabel("t$_{sat}$ (s)")
    ax2.set_ylim(0, 5)
    plt.yticks([1, 2, 3, 4])
    # ax2.legend(['Saturation duration'], loc="upper center",fontsize='x-large')
    ax2.yaxis.label.set_color(p2.get_color())
    ax2.yaxis.label.set_fontsize(fs)
    ax2.tick_params(axis="y", colors=p2.get_color(), labelsize=18)

    # set up the 3rd axis
    ax3 = ax1.twinx()
    # Offset the right spine of ax3.  The ticks and label have already been placed on the right by twinx above.
    ax3.spines.right.set_position(("axes", 1.2))
    # Plot line chart on axis #3
    (p3,) = ax3.plot(xaxis, Offsets_list, "C2")
    ax3.grid(False)  # turn off grid #3
    ax3.set_ylabel("$\delta$ (ppm)")
    ax3.set_ylim(-5, 6)
    # plt.yticks([0.5,1,1.5,2])
    # ax3.legend(['Saturation offset'], loc="upper right",fontsize='x-large')
    ax3.yaxis.label.set_color(p3.get_color())
    ax3.yaxis.label.set_fontsize(fs)

    ax3.tick_params(axis="y", colors=p3.get_color(), labelsize=18)

    ax1.set_xlabel("Acquisition points")

    fig.tight_layout
    plt.show()

    if saving:
        fig.savefig(savename, format="svg", dpi=660)
    return fig
