import corv
import pyphot
from pyphot import unit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pygaia.errors.astrometric import parallax_uncertainty
import pytorch_lightning as pl
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import truncnorm
import scipy.stats as ss

model = corv.models.Spectrum('1d_da_nlte')
library = pyphot.get_library()

def forward(teff, distance, radius, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP', 'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']):
    pc_to_m = 3.0856775814671916e16
    rsun_to_m = 6.957e8
    bands = [library[band] for band in bands]
    wavl, interp = model.wavl, model.model_spec
    flux = interp((teff, 8)) * ((radius*rsun_to_m) / (distance * pc_to_m))**2
    band_flux = np.array([band.get_flux(wavl * unit['AA'], flux * unit['erg/s/cm**2/AA'], axis=1).value for band in bands])
    return band_flux

def sim_forward(teff, distance, radius, bands):
    snr = np.random.uniform(300, 400)
    band_flux = forward(teff, distance, radius, bands)
    band_flux_noisy = np.random.normal(band_flux, band_flux/snr)

    gmag = -2.5 * np.log10(band_flux_noisy[0] / 2.4943e-09)
    plx_unc = parallax_uncertainty(gmag, release='dr3') * 1e-6
    plx = np.random.normal(loc = -0.000014 + (1 / distance), scale = plx_unc, size=(1))
    plx_data = np.concatenate([plx, np.array([plx_unc])])

    obs = np.concatenate([band_flux_noisy, band_flux_noisy/snr])
    return np.concatenate([plx_data, obs])

def simulate(n_train = 50_000, outpath = 'data', bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP', 'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']):
    # Simulate training data
    theta_samples = np.random.uniform(low=[2000, 5, 0.004], high=[80000, 400, 0.025], size=(n_train, 3))  # Parameter proposal
    x_samples = np.array([sim_forward(*theta, bands) for theta in tqdm(theta_samples)])

    np.save(os.path.join(outpath, 'wdparams_theta.npy'), theta_samples)
    np.save(os.path.join(outpath, 'wdparams_x.npy'), x_samples)

def load(path = 'data', dataloader = True, val_fraction = 0.1, batch_size = 128):
    theta_samples = torch.tensor(np.load(os.path.join(path, 'wdparams_theta.npy')), dtype=torch.float32)
    x_samples = torch.tensor(np.load(os.path.join(path, 'wdparams_x.npy')), dtype=torch.float32)

    if dataloader:
        # Normalize the data
        x_mean = x_samples.mean(dim=0)
        x_std = x_samples.std(dim=0)
        x_samples = (x_samples - x_mean) / x_std

        theta_mean = theta_samples.mean(dim=0)
        theta_std = theta_samples.std(dim=0)
        theta_samples = (theta_samples - theta_mean) / theta_std

        # construct dataloader
        n_samples_val = int(val_fraction * len(x_samples))
        dataset = TensorDataset(x_samples, theta_samples)
        dataset_train, dataset_val = random_split(dataset, [len(x_samples) - n_samples_val, n_samples_val])
        train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)

        return dataset_train, dataset_val, train_loader, val_loader, theta_mean, theta_std, x_mean, x_std
    else:
        return theta_samples, x_samples    


    