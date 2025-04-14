# Neural network and feature selection for quantitative CEST and acquisitions optimization

This is a python code for deep learning - based quantitative CEST, with a feature selection scheme proposed to shorten the acquisition schedule.

Associated commuication : Maguin C., Flament J., «Deep-learning for fast, specific quantitative CEST imaging», ISMRM Annual Meeting Proceedings, 2023 Toronto

## Summary

CEST quantification is achieved here through neural network prediction. We use a simple 2-layers neural network (300x300 nodes) and train it with realistic simulated CEST examples (MSE loss and RMSprop optimizer). We then propose to reduce the overly exhaustive acquisition schedule, so that future acquistiions are made shorter, by implementing a feature selection scheme (RFE) to select only the most relevant acquisition points. 

Here, the CEST dataset used for training was generated using analytic simulations for the sake of faster computing. The associated tools can be found in Simulation_tools folder. However, if you prefer to use fully numeric simulations, you should check out PyPulseq-CEST toolbox (https://pulseq-cest.github.io/) 

## Results and visualisation

----Need to update that----

_To do list :_
- _set up download for clean data_
- _do overview notebook with clean figures_
- _improve doc and clean outputs_

