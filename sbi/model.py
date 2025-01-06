import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning import LightningModule

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from torch.distributions import Chi2, transforms, transformed_distribution

import argparse
import pickle
import os

from . import simulator

def prior(theta, plx, e_plx, device, L = 350):
    """ implements priors for all three parameters,
    distance : truncated transformed chi2 prior with six degrees of freedom per Bailer-Jones 2015
    temperature & radius : uniform priors
    """
    # distance prior
    real_theta = theta * theta_std.to(device=device) + theta_mean.to(device=device)
    plx = plx * x_std[0].to(device=device) + x_mean[0].to(device=device)
    e_plx = e_plx * x_std[1].to(device=device) + x_mean[1].to(device=device)
    likelihood = torch.distributions.Normal(1/real_theta[:,1], e_plx).log_prob(plx)
    distance_prior = transformed_distribution.TransformedDistribution(
        Chi2(torch.tensor([6]).to(device=device)), 
        transforms.AffineTransform(loc=torch.tensor([0]).to(device=device), scale=torch.tensor([0.5 * L]).to(device=device))
    ).log_prob(real_theta[:,1])
    #uniform prior
    log_prior = torch.zeros(real_theta.shape[0], device=device)
    bounds = torch.tensor([[1000, 120000], [0, 2000], [0.001, 0.05]], device=device)
    for i in range(3):
        min_bound, max_bound = bounds[i]
        # Check if parameter i is within bounds for all examples
        within_bounds = (real_theta[:, i] >= min_bound) & (real_theta[:, i] <= max_bound)
        log_prior[~within_bounds] = torch.inf
    return likelihood + distance_prior + log_prior 

class NeuralPosteriorEstimator(LightningModule):
    """ Simple neural posterior estimator class using a normalizing flow as the posterior density estimator.
    """
    def __init__(self, featurizer_in, featurizer_h, featurizer_layers, nflow_h, nflow_layers, context_dimension):  
        super().__init__()
        self.hparam_dict = {'featurizer_in' : featurizer_in, 'featurizer_h' : featurizer_h, 'featurizer_layers' : featurizer_layers,
                        'nflow_h' : nflow_h, 'nflow_layers' : nflow_layers, 'context_dimension' : context_dimension}
        self.flow_in = 3
        
        """featurizer (simple multi-layer perceptron)
        """
        seq = [nn.Linear(featurizer_in, featurizer_h), nn.GELU()]
        for _ in range(featurizer_layers):
            seq += [nn.Linear(featurizer_h, featurizer_h), nn.GELU()]
        seq += [nn.Linear(featurizer_h, context_dimension)]
        self.featurizer = nn.Sequential(*seq)

        """normalizing flow model"""
        base_dist = StandardNormal(shape=[self.flow_in])
        transforms = []
        for _ in range(nflow_layers):
            transforms.append(ReversePermutation(features=self.flow_in))
            transforms.append(MaskedAffineAutoregressiveTransform(features=self.flow_in, hidden_features=nflow_h, context_features=context_dimension))
        transform = CompositeTransform(transforms)
        self.flow = Flow(transform, base_dist)

        # prior
        self.prior = prior

    def forward(self, x):
        return self.featurizer(x)
    
    def loss(self, x, theta):     
        plx, e_plx, distance = x[:,0], x[:,1],  theta[:,1]
        prior = self.prior(theta, plx, e_plx, self.device)
        context = self(x[:,2:])
        return - self.flow.log_prob(inputs=theta, context=context) - prior

    def training_step(self, batch, batch_idx):
        x, theta = batch
        loss = self.loss(x, theta).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, theta = batch
        loss = self.loss(x, theta).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)
    
    def save_dict(self, path):
        with open(path + '_hparams.pkl', 'wb') as hparams:
            pickle.dump(self.hparam_dict, hparams)

def load_model_from_path(path):
    with open(path + '_hparams.pkl', 'rb') as hparams:
        hparams = pickle.load(hparams)
    npe = NeuralPosteriorEstimator.load_from_checkpoint(f"{path}.ckpt", **hparams)
    return npe
    
if __name__ == "__main__":
    """nn parameters by component:
    - featurizer:
        hidden layer dimension (hidden_dim = 16)
        no. layers (n_layers = 2)
        output dimension (must be equal to d_context)
    - normalizing flow:
        hidden layer dimension (hidden_dim = 32)
        context dimensionality (d_context = 8)
        no. layers (n_layers = 2)
    """
    parser = argparse.ArgumentParser()
    # path variables
    parser.add_argument('--savepath', type=str)             # directory to save weights
    parser.add_argument('--datapath', type=str)             # directory to read simulated data
    # featurizer params
    parser.add_argument('--featurizer_h', nargs='?', const=16, type=int, default=16)
    parser.add_argument('--featurizer_layers', nargs='?', const=2, type=int, default=2)
    # normalizing flow params
    parser.add_argument('--nflow_h', nargs='?', const=32, type=int, default=32)
    parser.add_argument('--nflow_layers', nargs='?', const=2, type=int, default=2)
    parser.add_argument('--context_dimension', nargs='?', const=8, type=int, default=8)
    # data params
    parser.add_argument('--val_fraction', nargs='?', const=0.1, type=float, default=0.1)
    parser.add_argument('--batch_size', nargs='?', const=128, type=int, default=128)
    parser.add_argument('--max_epochs', nargs='?', const=35, type=int, default=35)
    args = parser.parse_args()

    # load data and build the model
    dataset_train, dataset_val, train_loader, val_loader, theta_mean, theta_std, x_mean, x_std = simulator.load(args.datapath, dataloader=True, 
                                                                                                                val_fraction = args.val_fraction,
                                                                                                                batch_size = args.batch_size)
    npe = NeuralPosteriorEstimator(x_mean.shape[0]-2, args.featurizer_h, args.featurizer_layers, args.nflow_h, args.nflow_layers, args.context_dimension)
    
    # start training
    trainer = Trainer(max_epochs=args.max_epochs)
    trainer.fit(model=npe, train_dataloaders=train_loader, val_dataloaders=val_loader);

    # save information
    print(f"save path: {args.savepath}.ckpt")
    trainer.save_checkpoint(f"{args.savepath}.ckpt")
    npe.save_dict(path = f"{args.savepath}")

