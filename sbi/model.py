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

class NeuralPosteriorEstimator(LightningModule):
    """ Simple neural posterior estimator class using a normalizing flow as the posterior density estimator.
    """
    def __init__(self, featurizer_in, featurizer_h, featurizer_layers, nflow_h, nflow_layers, context_dimension,
                 x_mean = None, x_std = None, theta_mean = None, theta_std = None):  
        super().__init__()
        self.hparam_dict = {'featurizer_in' : featurizer_in, 'featurizer_h' : featurizer_h, 'featurizer_layers' : featurizer_layers,
                        'nflow_h' : nflow_h, 'nflow_layers' : nflow_layers, 'context_dimension' : context_dimension}
        self.flow_in = 3

        self.x_mean = x_mean.to(self.device) if x_mean is not None else None
        self.x_std = x_std.to(self.device) if x_std is not None else None
        self.theta_mean = theta_mean.to(self.device) if theta_mean is not None else None
        self.theta_std = theta_std.to(self.device) if theta_std is not None else None
        
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

    def forward(self, x):
        return self.featurizer(x)
    
    def loss(self, x, theta):     
        plx, e_plx, distance = x[:,0], x[:,1],  theta[:,1]
        prior = self.prior(theta, plx, e_plx, self.device)
        context = self(x)
        return -self.flow.log_prob(inputs=theta, context=context) + prior
    
    def prior(self, theta, plx, e_plx, device, L = 1350):
        """ implements priors for all three parameters,
        distance : truncated transformed chi2 prior with six degrees of freedom per Bailer-Jones 2015
        temperature & radius : uniform priors
        """
        # unnormalize parameters
        real_theta = theta * self.theta_std.to(device=device) + self.theta_mean.to(device=device)
        plx = plx * self.x_std[0].to(device=device) + self.x_mean[0].to(device=device)
        e_plx = e_plx * self.x_std[1].to(device=device) + self.x_mean[1].to(device=device)
        # distance prior
        likelihood = -0.5*(torch.log(e_plx) + ((1000/real_theta[:,1]) - plx)**2/e_plx)
        distance_prior = -torch.log(real_theta[:,1]**2 / (2*L**3)) - real_theta[:,1] / L
        #uniform prior
        log_prior = torch.zeros(real_theta.shape[0], device=device)
        bounds = torch.tensor([[1000, 120000], [0, 2000], [0.004, 0.025]], device=device)
        for i in range(3):
            min_bound, max_bound = bounds[i]
            # Check if parameter i is within bounds for all examples
            within_bounds = (real_theta[:, i] >= min_bound) & (real_theta[:, i] <= max_bound)
            log_prior[~within_bounds] = -torch.inf
            #log_prior[within_bounds] = torch.tensor([0.0])
        return likelihood + distance_prior + log_prior 

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

def load_model_from_path(path, parameter_dict = None):
    with open(path + '_hparams.pkl', 'rb') as hparams:
        hparams = pickle.load(hparams)
    hparams = hparams | parameter_dict if parameter_dict is not None else hparams
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
    npe = NeuralPosteriorEstimator(x_mean.shape[0], args.featurizer_h, args.featurizer_layers, args.nflow_h, args.nflow_layers, args.context_dimension,
                                   theta_mean=theta_mean, theta_std=theta_std, x_mean=x_mean, x_std=x_std)
    
    # start training
    trainer = Trainer(max_epochs=args.max_epochs)
    trainer.fit(model=npe, train_dataloaders=train_loader, val_dataloaders=val_loader);

    # save information
    print(f"save path: {args.savepath}")
    trainer.save_checkpoint(f"{args.savepath}.ckpt")
    npe.save_dict(path = f"{args.savepath}")

