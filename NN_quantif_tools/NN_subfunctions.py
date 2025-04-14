# -*- coding: utf-8 -*-
"""
Deep learning models for qCEST 


Created on Wed Jul 20 16:34:30 2022

@author: Cecile Maguin
"""

# from analytic_simulation import analytic_simulation

import numpy as np
import torch as th
import time


class reconstruction_network(th.nn.Module):
    """Fully-connected 2 layers neural network - Everything is in Config so that its easy to do hyperparameter testing """

    def __init__(self, Config):

        super(reconstruction_network, self).__init__()

        self.Config = Config

        # Default is MSE loss
        self.loss_fn = th.nn.MSELoss()

        self.layer1 = th.nn.Sequential(
            th.nn.Dropout(self.Config["Dropout_rate"][0]),
            th.nn.Linear(self.Config["Input_size"], self.Config["Layer_size"][0]),
            th.nn.BatchNorm1d(self.Config["Layer_size"][0]),
            th.nn.Tanh(),
        )

        self.layer2 = th.nn.Sequential(
            th.nn.Dropout(self.Config["Dropout_rate"][1]),
            th.nn.Linear(self.Config["Layer_size"][0], self.Config["Layer_size"][1]),
            th.nn.BatchNorm1d(self.Config["Layer_size"][1]),
            th.nn.Tanh(),  # th.nn.ReLU()
        )

        self.output_layer = th.nn.Sequential(
            th.nn.Linear(self.Config["Layer_size"][1], self.Config["Output_size"]),
            th.nn.ReLU(),
        )

        self.epoch = 0

        self.Perf_record = {
            "Loss_record": [],
            "Valid_record": [],
            "Training_time": 0,
        }

    def forward(self, data):
        x = self.layer1(data)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x

    def score(self, X, y):
        return self.loss_fn(self.forward(X), y)


def train_model(
    model, X_training, X_validation, Y_training, Y_validation, Config, Nepochs,
):
    """Generic function to train pytorch model on training dataset and also evaluates performance on validation set along the way.
    Returns loss training history, validation error record, and total training time."""

    # Set print options for tensors
    th.set_printoptions(precision=3)

    # Set model to training state
    model.train()

    # Defining loss function and optimizer
    loss_fn = th.nn.MSELoss()
    # optimizer = th.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay )
    optimizer = th.optim.RMSprop(
        model.parameters(),
        lr=Config["Learning_rate"],
        weight_decay=Config["Weight_Decay"],
    )

    # Defining arrays to store values of loss and error
    Loss_training = np.zeros(Nepochs)
    Error_validation = np.zeros((Nepochs, model.Config["Output_size"]))

    # Mini-batching
    Training_size = X_training.shape[0]
    indices = np.arange(Training_size)  # index of training sequence
    minibatch_size = Config["Minibatch_Size"]
    n_batches = Training_size // minibatch_size

    print_frequency = max(1, Nepochs // 10)  # frequency to print follow up of training
    tic = time.perf_counter()  # for timing

    # Training
    for epoch in range(Nepochs):
        total_loss = 0.0
        np.random.shuffle(indices)

        for batch in range(n_batches):
            batch_ids = indices[batch * minibatch_size : (batch + 1) * minibatch_size]
            prediction = model(X_training[batch_ids, :])
            loss = loss_fn(prediction, Y_training[batch_ids, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        Loss_training[epoch] = total_loss

        # Also get score on validation set along the way
        model.eval()  # switch to eval state
        with th.no_grad():
            prediction_valid = model.forward(X_validation).cpu().detach().numpy()
            GT_valid = Y_validation.cpu().detach().numpy()
            error_valid = np.mean((prediction_valid - GT_valid) ** 2, axis=0)
            Error_validation[epoch, :] = error_valid
        # Print regularily some info on loss and validation to follow training
        if epoch % print_frequency == 0:
            toc = time.perf_counter()
            print(
                "Epoch #",
                model.epoch + epoch,
                " (elasped time is ",
                format(toc - tic, ".1f"),
                " s). Loss =",
                format(total_loss, ".3f"),
                " Validation error=",
                format(np.sum(np.sqrt(error_valid)), ".3f"),
                " \n",
            )
        model.train()
    model.epoch += Nepochs

    toc = time.perf_counter()
    timing = toc - tic

    # Add to model record the training we did here
    model.Perf_record["Loss_record"].append(Loss_training)
    model.Perf_record["Valid_record"].append(Error_validation)
    model.Perf_record["Training_time"] += timing

    model.eval()  # switch back to eval

    return Loss_training, Error_validation, timing


def feature_importance(model, X, y, feature_list=None, Shuffle_Avg=15):
    """Importance feature scoring. 
    Evaluates the score of all features specified in feature_list.
    If feature_list is not specified, then every feature is scored.
    Returns an array with the score of all features specified."""
    # If feature_list is not provided, use all features.
    if feature_list is None:
        feature_list = np.arange(0, model.D_in)
    # Compare permuted score to baseline score
    baseline_score = model.score(X, y).cpu().detach().numpy()
    permuted_scores = th.zeros(np.size(feature_list))

    for idx, feat in enumerate(feature_list):
        s = 0
        for k in range(Shuffle_Avg):  # To get robust scoring, average on a few shuffles
            s += get_score_after_permutation(model, X, y, feat)
        permuted_scores[idx] = s / Shuffle_Avg
    feature_scoring = permuted_scores - baseline_score
    return feature_scoring


def get_score_after_permutation(model, X, y, curr_feat):
    """ Returns the score of model when curr_feat is permuted randomly"""

    # Clone X to avoid modifying the original tensor.
    X_permuted = X.clone()

    # Generate a random permutation of row indices and permute the specified column.
    random_indices = th.randperm(X.size(0))
    X_permuted[:, curr_feat] = X_permuted[random_indices, curr_feat]

    permuted_score = model.score(X_permuted, y).cpu().detach().numpy()
    return permuted_score


def feature_elimination(model, X, y, n_features_to_select):
    """
    Feature Elimination.
    Eliminates the least important feature until the desired number 
    of features remains. In this case there is no re-training in between iterations
    
    Args:
        model: Pytorch model
        X (th.Tensor): Input features tensor.
        y (th.Tensor): Target values tensor.
        n_features_to_select (int): Number of features to select.
    
    Returns:
        best_features (list): List of indices of the remaining best features.
        removed_features (list): List of indices of features removed.
        scores (numpy array) : scores of the removed features
    """

    # Clone X to avoid modifying the original tensor.
    X_permuted_permanently = X.clone()
    n_features = model.Config["Input_size"]
    removed_features = []
    scores = np.zeros(n_features - n_features_to_select)

    # Eliminate features until the desired number is reached.
    for i in range(n_features - n_features_to_select):
        feature_score = feature_importance(
            model, X_permuted_permanently, y, np.arange(0, n_features, 1)
        )
        feature_score = feature_score.cpu().detach().numpy()

        # make sure not to select twice the same eliminated feature
        feature_score[removed_features] = 1e6
        # Find the least important features
        least_important_feature = np.argmin(feature_score)
        removed_features.append(least_important_feature)
        scores[i] = feature_score[least_important_feature]

        # Permanently permute the least important feature in X_permuted_permanently so that it won't affect subsequent scores
        X_permuted_permanently[:, least_important_feature] = X_permuted_permanently[
            th.randperm(X.size(dim=0)), least_important_feature
        ]
    # Remaining features are the ones not removed
    best_features = [
        x for x in np.arange(0, n_features, 1) if x not in removed_features
    ]

    return best_features, removed_features, scores
