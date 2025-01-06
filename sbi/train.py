from pytorch_lightning.trainer.trainer import Trainer
import argparse

from . import model
from . import simulator

def train_from_file(savepath, datapath, val_fraction, batch_size, max_epochs):
    dataset_train, dataset_val, train_loader, val_loader, theta_mean, theta_std, x_mean, x_std = simulator.load(datapath, dataloader=True, 
                                                                                                                val_fraction = val_fraction,
                                                                                                                batch_size = batch_size)
    parameter_dict = {"theta_mean" : theta_mean, "theta_std" : theta_std, "x_mean" : x_mean, "x_std" : x_std}
    npe = model.load_model_from_path(savepath, parameter_dict)    
    # start training
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(model=npe, train_dataloaders=train_loader, val_dataloaders=val_loader);
    # save information
    print(f"save path: {savepath}")
    trainer.save_checkpoint(f"{savepath}.ckpt")
    npe.save_dict(path = f"{savepath}")

def train_from_scratch(featurizer_in, featurizer_h, featurizer_layers, nflow_h, nflow_layers, context_dimension,
                       savepath, datapath, val_fraction, batch_size, max_epochs):
    # load data and build the model
    dataset_train, dataset_val, train_loader, val_loader, theta_mean, theta_std, x_mean, x_std = simulator.load(datapath, dataloader=True, 
                                                                                                                val_fraction = val_fraction,
                                                                                                                batch_size = batch_size)
    npe = model.NeuralPosteriorEstimator(x_mean.shape[0]-2, featurizer_h, featurizer_layers, nflow_h, nflow_layers, context_dimension,
                                         theta_mean=theta_mean, theta_std=theta_std, x_mean=x_mean, x_std=x_std)
    # start training
    trainer = Trainer(max_epochs=max_epochs)
    trainer.fit(model=npe, train_dataloaders=train_loader, val_dataloaders=val_loader);
    # save information
    print(f"save path: {savepath}")
    trainer.save_checkpoint(f"{savepath}.ckpt")
    npe.save_dict(path = f"{savepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path variables
    parser.add_argument('--savepath', type=str)             # directory to save weights
    parser.add_argument('--datapath', type=str)             # directory to read simulated data
    parser.add_argument('--trainfrom', type=str, default='scratch')
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

    assert args.trainfrom in ['scratch', 'file']
    if args.trainfrom == 'file':
        train_from_file(args.savepath, args.datapath, args.val_fraction, args.batch_size, args.max_epochs)
    elif args.trainfrom == 'scratch':
        train_from_scratch(args.featurizer_in, args.featurizer_h, args.featurizer_layers, args.nflow_h, args.nflow_layers, args.context_dimension,
                       args.savepath, args.datapath, args.val_fraction, args.batch_size, args.max_epochs)