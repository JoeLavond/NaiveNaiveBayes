""" Packages """
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal

# source files
import nnb_utils as utils


""" HELPER FUNCTIONS
Result:
    Both helper functions return the weighted versions of column means and standard deviations
    If no weights are given, they will assume each observation has equal weighting
    Inputs are assumed to be of type torch.Tensor
Usage:
    Probability estimates for one-hot encoded columns are the mean of the indicator column
    Need weighted mean when label column including soft labeling, for semi-supervised learning
    For continous variables, mean and standard deviation are needed to specify Normal distribution
"""        
def col_means(x, row_weights=None):
    out = x.float()  # mean operation requires float type
    
    if row_weights is not None:
        out = out[row_weights.nonzero().squeeze()]  # need to remove observations given no weight
        out = out * row_weights[row_weights.nonzero().squeeze()].view(-1, 1)
        return out.sum(dim=0) / row_weights.sum()
    
    return out.mean(dim=0) 
    
def col_stds(x, row_weights=None):
    out = x.float()
    
    if row_weights is not None: 
        means = col_means(out, row_weights)
        
        # compute weighted standard deviation
        out = out[row_weights.nonzero().squeeze()]
        out = (out - means) ** 2
        out = out * row_weights[row_weights.nonzero().squeeze()].view(-1, 1)
        
        m = row_weights.nonzero().sum()  # scale by how many observations had any weight
        return torch.sqrt(
            out.sum(dim=0) / ((m - 1) / m * row_weights.sum())
        )
        
    return out.std(dim=0) 

    
""" HELPER LAYERS
Return torch model layers with key functionalities needed for model implmentation
"""
class StdLayer(nn.Module):
    """
    Function:
        Return model layer that standardizes input data   
    Usage:
        Standardize input continuous data and store training mean and standard deviationn
        Use store training summary statistics to scale new input data for prediction
    """
    
    def __init__(self, x_cts):
        super(StdLayer, self).__init__()
        
        # setup - store training summary statistics
        self.register_buffer(
            'mean', 
            col_means(x_cts).view(1, -1)
        )
        self.register_buffer(
            'std', 
            col_stds(x_cts).view(1, -1)
        )
        
    def forward(self, x):
        return (x - self.mean) / self.std 
    

class CatLayer(nn.Module):
    """
    Function:
        Model layer that calulates categorical lifts for both y = 0 and y = 1, typically y = 0 not of interest
        Will perform feature selection; mask_ind = 1 given one of the following:
            1. p = proportion of column lifts to use
            2. k = number of column lifts to use
            3. v = how far from 1 must a lift be to be used?
                ex. v = 0.23 -> lifts outside of .77 and 1.23 will be used
        Note lower_lim and upper_lim prevent one lift of zero (or inf) from dominating the prediction
    """
    
    def __init__(
        self, 
        x_cat, y,   # --------------- input data 
        mask_ind, p, k, v,  # ------- feature selection
        lower_lim, upper_lim  # ----- thresholding
    ):
        super(CatLayer, self).__init__()
        
        # setup - store function options: features selection and thresholding
        self.lower_lim, self.upper_lim = lower_lim, upper_lim
        self.mask_ind = mask_ind
        
        # categorical lifts: P(X = x | Y = y) / P(X = x) for each y
        self.register_buffer(
            'cat_lifts', 
            torch.max(
                torch.stack([
                    col_means(x_cat, row_weights=(1 - y.float())),
                    col_means(x_cat, row_weights=y)
                ], dim=0) / col_means(x_cat), 
                torch.tensor(lower_lim)
            )
        )
        
        # store masking vector for lift feature selection
        if self.mask_ind:
            self.mask_vector = self.get_forward_mask_(p, k, v)
        
        
    def get_forward_mask_(self, p=None, k=None, v=None):
        """
        Function: 
            Will perform feature selection; mask_ind = 1 given one of the following:
                1. p = proportion of column lifts to use
                2. k = number of column lifts to use
                3. v = how far from 1 must a lift be to be used?
                    ex. v = 0.23 -> lifts outside of .77 and 1.23 will be used
            Note specifying p and k, will return MORE features when ties exist!
        Usage:
            Removing columns with lift close to 1 will greatly improve model performance
                Bayesian lift factors near one greatly increase model variability with little bias improvment
                Current recomendation is to tune feature selection on training datasets
                 ex. grid or exhausive search with using some performance criterion
            Data reduction prior to other algorithms
        """
        sub_lifts = self.cat_lifts[1]
        values = (1 - sub_lifts).abs()
        
        # if not value size masking
        if v is None:
            
            # determine number of unmasked lifts
            if p is not None:
                assert 0 <= p and p <= 1
                k = int(p * len(values))  # if p given, use to find k
                
            sorted_values, _ = values.sort(descending=True)
            v = sorted_values[min(k, len(sorted_values)) - 1].item()  # keep the k largest lifts
            
        # set mask vector
        self.unmasked_vector = (values >= v)
        sub_lifts = sub_lifts[self.unmasked_vector]
        
        # print masking information - what range of lifts are not present?
        try:
            l = sub_lifts[sub_lifts <= 1].max().item()
        except:
            l = 0
        try:
            u = sub_lifts[sub_lifts >= 1].min().item()
        except:
            u = np.inf
            
        print(
            f'LIFT INFORMATION - Count: {self.unmasked_vector.sum().item():d},'
            + f' Masked Between: {l:.4f}'
            + f' And {u:.4f}'
        )
        
    
    def forward(self, x, c=1):
        """
        Remarks:
            Currently relies on matrix multiplication of input and lifts, room for improvement
                Could use indexing if single prediction
                Could index away columns not needed in batch prediction?
            Masking could be used earlier in function to reduce math operations
        """

        # index lifts exhibited by observation x's dummy coding, replace others with 1 
        out = x * self.cat_lifts[c] 
        out[x == 0] = 1
        
        # use thresholding to keep single column from dominating prediction
        out = torch.max(out, torch.tensor(self.lower_lim))
        out = torch.min(out, torch.tensor(self.upper_lim))
        
        # mask unwanted lifts
        if self.mask_ind:
            out = out[..., self.unmasked_vector]
            
        out = torch.prod(out, dim=1)
        return out
    
    
class CtsLayer(nn.Module):
    """
    Function:
        Model layer that calulates continuous lifts for both y = 0 and y = 1, typically y = 0 not of interest
            Fits Normal distribution through weighted mean and standard deviation of input column data
            Note that general population is Normal(0, 1) as continuous data is 0-1 standardized
                Allows for reduced storage
        CANNOT implement same feature selection here
            Lifts change dependent on values of inputs to prediction
        Note lower_lim and upper_lim prevent one lift of zero (or inf) from dominating the prediction
    """
    
    def __init__(self, 
        x_cts, y,  # ---------------- input training data
        lower_lim, upper_lim  # ----- thresholding
    ):
        super(CtsLayer, self).__init__()
        
        # setup - create general population distribution and store thresholding info
        self.std_norm = Normal(0, 1)
        self.lower_lim, self.upper_lim = lower_lim, upper_lim
        
        # store training data subpopulation means and standard deviations for distribution use
        self.means = torch.stack([
            col_means(x_cts, row_weights=(1 - y.float())),
            col_means(x_cts, row_weights=y)
        ], dim=0)
        self.stds = torch.stack([
            col_stds(x_cts, row_weights=(1 - y.float())),
            col_stds(x_cts, row_weights=y)
        ], dim=0)
        
        
    def forward(self, x, c=1):
        
        # Create distribution from training column statitics and evaluate density at x values
        x_norm = torch.stack(
            [
                torch.exp(
                    Normal( 
                        self.means[c, i], 
                        self.stds[c, i], 
                    ).log_prob(  
                        x[:, i]  
                    )
                ) for i in range(x.size(-1))
            ], dim=1
        )
        
        # General population all N(0, 1) - as input data is standardized
        x_std_norm = torch.exp(
            self.std_norm.log_prob(x.reshape(-1))
        )
        x_std_norm = x_std_norm.view(x.size())
        
        # Return lifts for each feature and new x
        out = torch.max(
            x_norm / x_std_norm, 
            torch.tensor(self.lower_lim)
        )
        out = torch.nan_to_num(out, nan=self.upper_lim, posinf=self.upper_lim)  # need to handle division by 0 
        
        out = torch.prod(out, dim=1)
        return out
   
    
""" MAIN MODEL IMPLEMENTATION 
Notation:
    Prefixes:
        l = labeled data
        u = unlabeled data
        x = input data
        y = output data, must be 0-1 encoding, only support for 2-way classification problem
    Suffixes:
        _cat = one-hot encoding matrix for categorical data or discretized continous data
        _cts = continous data, will be 0-1 standardized by model
"""
class NaiveNaiveBayes(nn.Module):

    """
    Returns:
        Assumes input is numpy array
        Will perform feature selection given one of the following:
            1. p = proportion of column lifts to use
            2. k = number of column lifts to use
            3. v = how far from 1 must a lift be to be used?
                ex. v = 0.23 -> lifts outside of .77 and 1.23 will be used
        Model will threshold any lift values produced in between lower_ and upper_limit
        Can choose classification cutoff based on geometric mean or f1 score
        Can have model output diagnostic visuals (ROC and PR curves)
    Usage:
        Return NNB model initialized to LABELED training data
        Can update model using unlabeled data
    """

    def __init__(
        self, 
        ly, lx_cat=None, lx_cts=None, # ------- labeled data
        ux_cat=None, ux_cts=None,  # ---------- unlabeled data
        p=None, k=None, v=None,  # ------------ feature selection strategies
        lower_lim=0.01, upper_lim=100,  # ----- thresholding: floor/ceiling for any features's lift
        criterion='g', visual=1  # ------------ criteria for deciding model cutoff
    ):
        
        """ Setup """
        # Check inputs
        assert lx_cat is not None or lx_cts is not None, 'need some labeled input'
        assert criterion in ('g', 'f'), 'choose threshold based on g = sqrt(tpr * (1 - fpr)) or f = 2 * (precision * recall) / (precision + recall)'
        
        # given exactly one of p, k, and v - set mask_ind = 1 and perform feature selection on _cat variables
        self.mask_ind, self.visual = 0, visual
        self.p, self.k, self.v = p, k, v
        if p is not None or k is not None or v is not None:
            self.mask_ind = 1
            assert (p is not None) + (k is not None) + (v is not None) == 1, 'specify max one masking strategy'
            assert lx_cts is None and ux_cts is None, 'can use lift masking for data with dummy coding variables only'

        # save labels and thresholding
        super(NaiveNaiveBayes, self).__init__()
        self.lower_lim, self.upper_lim = lower_lim, upper_lim
        self.ly = torch.from_numpy(ly)

        """ Initialize input data """
        # Checks for valid entries and convert to tensor
        if ux_cat is not None:
            ux_cat = torch.from_numpy(ux_cat)
        if ux_cts is not None:
            ux_cts = torch.from_numpy(ux_cts) 
        
        # combine labeled and unlabeled data
        if lx_cat is not None:
            n_total = len(lx_cat)
            self.lx_cat = torch.from_numpy(lx_cat)
            self.x_cat = (torch.cat([self.lx_cat, ux_cat], dim=0) if ux_cat is not None else self.lx_cat).bool()
        else:
            self.lx_cat, self.x_cat = None, None
            
        if lx_cts is not None:
            
            n_total = len(lx_cts)
            self.lx_cts = torch.from_numpy(lx_cts)
            self.x_cts = (torch.cat([self.lx_cts, ux_cts], dim=0) if ux_cts is not None else self.lx_cts)
            
            self.std_layer = StdLayer(self.x_cts)
            self.x_cts = self.std_layer(self.x_cts)
            
        else:
            self.lx_cts, self.x_cts = None, None
        
        
        """ Initialize NNB model from labeled data """
        y = torch.zeros(n_total).bool()
        y[:len(self.ly)] = self.ly
        
        self.base = torch.tensor([  # get base rate of occurance for y=1 - only calc from known truth values (ly)
            torch.mean((1 - self.ly.float())).item(), 
            torch.mean(self.ly.float()).item()  
        ])
        self.get_model_layers_(row_weights=y, base=0)  # create model layers to calculate lifts - do not recalc base!
        self.criterion = criterion
        self.get_model_thresh_(self.criterion, self.visual)  # choose cutoff for pred=1

        
    def get_model_layers_(self, row_weights, base=1):
        """
        Returns:
            Compute new base rate only for semi-supervised training
                For initialization base rate must be computed only from labeled data
                0's for unlabed data will lead to falsely small base rate
            Note that unsupervised data will be used to make probability estimates for the entire population
        """
        
        if base:  # should base-rate be re-calc from row_weights?
            self.base = torch.tensor([  
                torch.mean((1 - row_weights.float())).item(), 
                torch.mean(row_weights.float()).item()  
            ])

        # init model layers for lift calculations
        if self.x_cts is not None:
            self.cts_layer = CtsLayer(self.x_cts, row_weights, self.lower_lim, self.upper_lim) 
            
        if self.x_cat is not None:
            self.cat_layer = CatLayer(
                self.x_cat, row_weights, 
                self.mask_ind, self.p, self.k, self.v, 
                self.lower_lim, self.upper_lim
            )
        
        
    def get_model_thresh_(self, criterion, visual=0):
        """
        Returns
            Determine cutoff for pred=1 using criterion
            Make ROC and PR plots
            Note that criterion choosen only based on labeled data
        """

        # roc
        preds = self(self.lx_cat, self.lx_cts)
        
        fpr, tpr, ft_thresh = roc_curve(
            y_true=self.ly, 
            y_score=preds
        )    
        
        precision, recall, pr_thresh = precision_recall_curve(
            y_true=self.ly, 
            probas_pred=preds
        )
        precision, recall = precision[:-1], recall[:-1]
        
        # plot roc and precision recall curves
        if visual:
            RocCurveDisplay.from_predictions(
                self.ly, 
                self(self.lx_cat, self.lx_cts)
            )
            PrecisionRecallDisplay.from_predictions(
                self.ly, 
                self(self.lx_cat, self.lx_cts)
            )
        
        # choose threshold
        if criterion == 'g':
            score = np.sqrt(tpr + (1 - fpr))  # Geometric Mean
            thresh = ft_thresh
        else:
            score = 2 * precision * recall / (precision + recall)  # F1 Score
            thresh = pr_thresh
            
        value, index = np.nanmax(score), np.nanargmax(score)
        self.thresh = thresh[index]
        print(f'THRESHOLD INFORMATION - Thresh: {self.thresh:.10f}, Criterion: {criterion}, Score: {value:.3f}')
                

    def update_model_(self, visual=0):
        """ 
        Returns:
            Re-initialize model using soft labels for unlabeled data given by existing model predictions
        Usage:
            Update model based on semi-supervised data
            Repeat until termination criteria
                Recommend fixed number of iterations
                Often model can cycle between several states and not terminate reduction criteria
        """
        
        # get model output and define threshold
        output = self(self.x_cat, self.x_cts)
        self.get_model_thresh_(self.criterion)
        preds = (output >= self.thresh)
        
        # use known truth where possible instead of soft classification - optional
        preds[:len(self.ly)] = self.ly
        
        # update model layers using soft labels for unlabeled data
        self.get_model_layers_(preds)
        self.get_model_thresh_(self.criterion, visual)
        
        
    def predict(self, x_cat=None, x_cts=None, y=None, print_all=1):
        """
        Returns:
            Allows for model prediction from numpy array
            Prints or returns model accuracy when given truth labels instead of output
        """
        
        # setup
        if x_cat is not None:
            x_cat = torch.from_numpy(x_cat)
        if x_cts is not None:
            x_cts = torch.from_numpy(x_cts)
            
        # make predicitons
        out = self(x_cat, x_cts)
        preds = (out >= self.thresh).numpy()
        
        # print accuracy if truth labels given
        if y is not None:
            acc = np.mean(y == preds)
            if print_all:
                print(f'Labeled Accuracy: {acc:.4f}')
                utils.print_cm(y, preds)
            return acc
            
        return out
    
        
    def forward(self, x_cat, x_cts, c=1):
        """ Returns: Prediction for torch.Tensor objects """
        
        if self.x_cts is not None:
            x_cts = self.std_layer(x_cts)  # standardize
            cts_lifts = self.cts_layer(x_cts, c) 
        else:
            cts_lifts = 1

        # calculate lifts
        cat_lifts = self.cat_layer(x_cat, c) if self.x_cat is not None else 1
        
        return self.base[c] * cts_lifts * cat_lifts