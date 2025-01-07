import corv
import pyphot
from pyphot import unit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from pygaia.errors.astrometric import parallax_uncertainty
import pytorch_lightning as pl
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.stats import truncnorm
import scipy.stats as ss
import pyvo
from astroquery.gaia import Gaia

model = corv.models.Spectrum('1d_da_nlte')
library = pyphot.get_library()

def get_ngf21():
    tap_service = pyvo.dal.TAPService("http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")
    QUERY = f"""select top 10000 GaiaDR3 as source_id, Plx, e_Plx
            from \"J/MNRAS/508/3877/maincat\" as ngf
            join \"J/A+A/674/A33/gspc-wd\" as gspc 
            on ngf.GaiaEDR3 = gspc.GaiaDR3
            where ngf.e_TeffH is not NULL and ngf.e_loggH is not NULL and ngf.e_TeffHe is not NULL and ngf.e_loggHe is not NULL
            and RAND() < 0.01"""
    ngfwds = tap_service.search(QUERY).to_table().to_pandas()
    gaiaquery = f"""select source_id, r_med_geo
                from external.gaiaedr3_distance
                where source_id in {tuple(ngfwds.source_id)}"""
    gaiadists = Gaia.launch_job_async(gaiaquery).get_results().to_pandas();
    return pd.merge(ngfwds, gaiadists, on="source_id")[['Plx', 'e_Plx', 'r_med_geo']]

def get_plx_data():
    ngf21 = get_ngf21()
    data = np.array([ngf21.Plx, ngf21.e_Plx, ngf21.r_med_geo]).T
    return data

class Simulator:
    def __init__(self, bands = ['Gaia_G', 'Gaia_BP', 'Gaia_RP', 'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']):
        ngf21 = get_ngf21()
        self.plxdata = np.array([ngf21.Plx, ngf21.e_Plx, ngf21.r_med_geo]).T
        self.plxdata = self.plxdata[self.plxdata[:,0].argsort()]
        self.bands = bands

    def err_from_plx(self, plx):
        return np.interp(plx, self.plxdata[:,0], self.plxdata[:,1])

    def forward_noiseless(self, teff, distance, radius):
        pc_to_m = 3.0856775814671916e16
        rsun_to_m = 6.957e8
        wavl, interp = model.wavl, model.model_spec
        flux = interp((teff, 8)) * ((radius*rsun_to_m) / (distance * pc_to_m))**2
        band_flux = np.array([library[band].get_flux(wavl * unit['AA'], flux * unit['erg/s/cm**2/AA'], axis=1).value for band in self.bands])
        plx = np.array([1000/distance])
        return np.concatenate([plx, band_flux])

    def forward_noisy(self, teff, distance, radius):
        snr = np.random.uniform(300, 400)
        noiseless = self.forward_noiseless(teff, distance, radius)
        plx, flux = noiseless[0], noiseless[1:]
        # compute noisy fluxes from the drawn SNR
        err_flux = flux / snr
        flux_noisy = np.random.normal(flux, err_flux)
        # add the noise to the simulated parameters
        gmag = -2.5 * np.log10(flux_noisy[0] / 2.4943e-09)
        err_plx = parallax_uncertainty(gmag, release='dr3') * 1e-3
        plx_noisy = np.random.normal(plx, err_plx)
        # combine the two datatypes then send them
        plxobs = np.array([plx_noisy, err_plx])
        fluxobs = np.concatenate([flux_noisy, err_flux])
        return np.concatenate([plxobs, fluxobs])

    def __call__(self, n_train = 50_000, outpath = 'data'):
        # Simulate training data
        temperature = np.random.uniform(low=2000, high=80000, size=n_train)
        dist_indx = np.random.randint(0, len(self.plxdata), size=n_train)
        radius = np.random.uniform(low=0.004, high=0.025, size=n_train)
        theta_samples = np.array([temperature, self.plxdata[dist_indx,-1], radius]).T
        # simulate noisy data
        x_samples = np.array([self.forward_noisy(*theta) for theta in tqdm(theta_samples)])
        # save the data
        np.save(os.path.join(outpath, 'wdparams_theta.npy'), theta_samples)
        np.save(os.path.join(outpath, 'wdparams_x.npy'), x_samples)
        return theta_samples, x_samples

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


    