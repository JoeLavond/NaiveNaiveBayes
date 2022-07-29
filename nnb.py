""" Packages """
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


""" Helper functions """
def col_means(x, row_weights=None):
    out = x.float()

    if row_weights is not None:
        out = out[row_weights.nonzero().squeeze()]
        out = out * row_weights[row_weights.nonzero().squeeze()].view(-1, 1)
        return out.sum(dim=0) / row_weights.sum()

    return out.mean(dim=0)

def col_stds(x, row_weights=None):
    out = x.float()

    if row_weights is not None:
        means = col_means(out, row_weights)

        out = out[row_weights.nonzero().squeeze()]
        out = (out - means) ** 2
        out = out * row_weights[row_weights.nonzero().squeeze()].view(-1, 1)

        m = row_weights.nonzero().sum()
        return torch.sqrt(
            out.sum(dim=0) / ((m - 1) / m* row_weights.sum())
        )

    return out.std(dim=0)


""" Helper Layers """
class StdLayer(nn.Module):

    def __init__(self, x_cts):
        super(StdLayer, self).__init__()

        # setup
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

    def __init__(
        self,
        x_cat, y,
        mask_ind, p, k, v,
        lower_lim, upper_lim
    ):
        super(CatLayer, self).__init__()

        # setup
        self.lower_lim, self.upper_lim = lower_lim, upper_lim
        self.mask_ind = mask_ind

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

        if self.mask_ind:
            self.mask_vector = self.get_forward_mask_(p, k, v)


    def get_forward_mask_(self, p=None, k=None, v=None):
        """ Note v = how far from one does a lift value need to be to keep """
        sub_lifts = self.cat_lifts[1]
        values = (1 - sub_lifts).abs()

        # if not value size masking
        if v is None:

            # determine number of unmasked lifts
            if p is not None:
                assert 0 <= p and p <= 1
                k = int(p * len(values))

            sorted_values, _ = values.sort(descending=True)
            v = sorted_values[min(k, len(sorted_values)) - 1].item()

        # set mask vector
        self.unmasked_vector = (values >= v)
        sub_lifts = sub_lifts[self.unmasked_vector]

        # get lower/upper lift bounds
        try:
            l = sub_lifts[sub_lifts <= 1].max().item()
        except:
            l = -1 * np.inf
        try:
            u = sub_lifts[sub_lifts >= 1].max().item()
        except:
            u = np.inf

        print(
            f'LIFT INFORMATION - Count: {self.unmasked_vector.sum().item():d},'
            + f' Masked Between: {l:.1f}'
            + f' And {u:.1f}'
        )


    def forward(self, x, c=1):

        out = x * self.cat_lifts[c]
        out[x == 0] = 1
        out = torch.max(out, torch.tensor(self.lower_lim))
        out = torch.min(out, torch.tensor(self.upper_lim))

        # mask unwanted lifts
        if self.mask_ind:
            out = out[..., self.unmasked_vector]

        out = torch.prod(out, dim=1)
        return out


class CtsLayer(nn.Module):

    def __init__(self, x_cts, y, lower_lim, upper_lim):
        super(CtsLayer, self).__init__()

        # setup
        self.std_norm = Normal(0, 1)
        self.lower_lim, self.upper_lim = lower_lim, upper_lim

        self.means = torch.stack([
            col_means(x_cts, row_weights=(1 - y.float())),
            col_means(x_cts, row_weights=y)
        ], dim=0)
        self.stds = torch.stack([
            col_stds(x_cts, row_weights=(1 - y.float())),
            col_stds(x_cts, row_weights=y)
        ], dim=0)


    def forward(self, x, c=1):

        # Create distribution from column statitics and evaluate density at x values
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

        # General population all N(0, 1)
        x_std_norm = torch.exp(
            self.std_norm.log_prob(x.reshape(-1))
        )
        x_std_norm = x_std_norm.view(x.size())

        # Return lifts for each feature and new x
        out = torch.max(
            x_norm / x_std_norm,
            torch.tensor(self.lower_lim)
        )
        out = torch.nan_to_num(out, nan=self.upper_lim, posinf=self.upper_lim)

        out = torch.prod(out, dim=1)
        return out


""" Main Model """
class NaiveNaiveBayes(nn.Module):

    """
    Inputs:
    - ?x_cat matrix of one-hot vectors
    - ?x_cts matrix of approximately continuous variables
    - ly one-hot vector of true labels

    Remarks:
    a. Currently ONLY support 2-class classification
    b. Model cannot handle indicator variables for categories not seen before
        - If a new value in seen for a ordinal varible,
            truncate to most extreme seen category
        - If a new value is seen for a categorical variable,
            give equal probability to all seen levels

    Results:
    a. Given ONLY labeled data, create NNB model
    b. Given BOTH labeled and unlabeled data
        - Initialize NNB model based on labeled data
        - Repeat until convergence
            0. Terminate if threshold does not change enough
            1. Perform soft classification of unlabeled data
            2. Use hard and soft labeled data to update NNB model
    """

    def __init__(
        self,
        ly,
        lx_cat=None, lx_cts=None, # labeled data
        ux_cat=None, ux_cts=None,  # unlabeled data
        p=None, k=None, v=None,  # feature selection strategies
        lower_lim=0.01, upper_lim=100,  # floor/ceiling for any features's lift
        criterion='g', visual=1  # choosing threshold
    ):

        # Check inputs
        assert lx_cat is not None or lx_cts is not None, 'need some labeled input'
        assert criterion in ('g', 'f'), 'choose threshold based on g = sqrt(tpr * (1 - fpr)), f = 2 * (precision * recall) / (precision + recall)'

        self.mask_ind, self.visual = 0, visual
        self.p, self.k, self.v = p, k, v
        if p is not None or k is not None or v is not None:
            self.mask_ind = 1
            assert (p is not None) + (k is not None) + (v is not None) == 1, 'specify max one masking strategy'
            assert lx_cts is None and ux_cts is None, 'can use lift masking for data with dummy coding variables only'

        super(NaiveNaiveBayes, self).__init__()
        self.lower_lim, self.upper_lim = lower_lim, upper_lim
        self.ly = torch.from_numpy(ly)

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

        self.base = torch.tensor([  # only calc from known truth values
            torch.mean((1 - self.ly.float())).item(),
            torch.mean(self.ly.float()).item()
        ])
        self.get_model_layers_(row_weights=y, base=0)
        self.criterion = criterion
        self.get_model_thresh_(self.criterion, self.visual)


    def get_model_layers_(self, row_weights, base=1):

        if base:  # should base-rate be re-calc from row_weights?
            self.base = torch.tensor([
                torch.mean((1 - row_weights.float())).item(),
                torch.mean(row_weights.float()).item()
            ])

        # calculate lifts
        if self.x_cts is not None:
            self.cts_layer = CtsLayer(self.x_cts, row_weights, self.lower_lim, self.upper_lim)

        if self.x_cat is not None:
            self.cat_layer = CatLayer(
                self.x_cat, row_weights,
                self.mask_ind, self.p, self.k, self.v,
                self.lower_lim, self.upper_lim
            )


    def get_model_thresh_(self, criterion, visual=0):

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


    def update_model_(self, criterion, visual=0):
        """ For use with semi-supervised data only """

        # get model output and define threshold
        output = self(self.x_cat, self.x_cts)
        self.get_model_thresh_(criterion)
        preds = (output >= self.thresh)

        # use known truth where possible instead of soft classification
        preds[:len(self.ly)] = self.ly

        # update model layers using soft labels for unlabeled data
        self.get_model_layers_(preds)
        self.get_model_thresh_(criterion, visual)


    def predict(self, x_cat=None, x_cts=None, y=None, print_all=1):

        # setup
        if x_cat is not None:
            x_cat = torch.from_numpy(x_cat)
        if x_cts is not None:
            x_cts = torch.from_numpy(x_cts)

        # make predicitons
        out = self(x_cat, x_cts)
        preds = (out >= self.thresh).numpy()

        if y is not None:
            acc = np.mean(y == preds)
            if print_all:
                print(f'Labeled Accuracy: {acc:.4f}')
                print_cm(y, preds)
            return acc

        return out


    def forward(self, x_cat, x_cts, c=1):

        if self.x_cts is not None:
            x_cts = self.std_layer(x_cts)  # standardize
            cts_lifts = self.cts_layer(x_cts, c)
        else:
            cts_lifts = 1

        # calculate lifts
        cat_lifts = self.cat_layer(x_cat, c) if self.x_cat is not None else 1

        return self.base[c] * cts_lifts * cat_lifts


