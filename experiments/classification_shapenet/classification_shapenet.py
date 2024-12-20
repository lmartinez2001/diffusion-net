"""
Widely inspired from classification_shrec11 package
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from shapenet_dataset import ShapenetDataset

# logging
from comet_ml import Experiment

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# === Arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_features", type=str, help="What features to use as input ('xyz' or 'hks') default: hks", default = 'xyz')
parser.add_argument("--diffusion_blocks", type=int, help="Number DiffusionNet Blocks", default=8)
parser.add_argument("--diffusion_features", type=int, help="Number of internal feature channels in DiffusionNet", default=64)
parser.add_argument("--n_class", type=int, help="Number of classes to predict", default=50)
parser.add_argument("--k_eig", type=int, help="Size of eigendecomposition", default=128)
parser.add_argument("--device", type=int, help="Gpu to use", default=0)
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
parser.add_argument("--epochs", type=int, help="number of epochs", default=200)
parser.add_argument("--save_dir", type=str, help="Save model directory", default='saved_models')
parser.add_argument("--comet_experiment", type=str, help="Name of comet experiment", default='experiment')
args = parser.parse_args()

        
# Intialize cometml logging
with open('private.yml', 'r') as f:
    private = yaml.load(f, Loader=yaml.SafeLoader)
logger = Experiment(
    api_key=private['comet']['key'],
    project_name=private['comet']['project'],
    workspace=private['comet']['workspace'],
)
logger.set_name(args.comet_experiment)
logger.log_parameters(vars(args))

# Create save dir for checkpoints
if not os.path.isdir(args.save_dir): os.mkdir(args.save_dir)

device = torch.device(f'cuda:{args.device}')
dtype = torch.float32
n_class = args.n_class
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = args.k_eig

# Training parameters
n_epoch = args.epochs
lr = args.lr
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')
label_smoothing_fac = 0.2

base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, 'op_cache')


# === Load datasets
# Train dataset
train_dataset = ShapenetDataset(split='train', k_eig=k_eig, op_cache_dir=op_cache_dir)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# Test dataset
val_dataset = ShapenetDataset(split='val', k_eig=k_eig, op_cache_dir=op_cache_dir)
val_loader = DataLoader(val_dataset, batch_size=None)


# === Create the model
C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features

model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                          C_out=n_class,
                                          C_width=args.diffusion_features, 
                                          N_block=args.diffusion_blocks, 
                                          last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                          outputs_at='global_mean', 
                                          dropout=True)

model = model.to(device)


# === Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(epoch):
    global lr 
    if epoch > 0 and epoch % decay_every == 0:
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
        logger.log_metric('learning_rate', lr, step=epoch)

    model.train()
    optimizer.zero_grad()
    
    correct = 0
    total_num = 0
    total_loss = 0
    for data in tqdm(train_loader):
        path, verts, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points_y(verts)
            # verts = diffusion_net.utils.random_rotate_points(verts)
        
        verts = verts.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        loss = diffusion_net.utils.label_smoothing_log_loss(preds, labels, label_smoothing_fac)
        loss.backward()
        total_loss += loss.item()
        
        pred_labels = torch.max(preds, dim=-1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        correct += this_correct
        total_num += 1

        optimizer.step()
        optimizer.zero_grad()
    
    total_loss /= total_num
    train_acc = correct / total_num
    logger.log_metric('train_loss', train_loss, step=epoch)
    logger.log_metric('train_acc', train_acc, step=epoch)
    return train_acc


def val():    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            path, verts, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            verts = verts.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            correct += this_correct
            total_num += 1

    val_acc = correct / total_num
    logger.log_metric('val_acc', val_acc, step=epoch)
    return val_acc 


best_val = -1
print("Training...")
for epoch in range(n_epoch):
    train_acc = train_epoch(epoch)
    print(f'Epoch {epoch} - Train overall: {100*train_acc:.2f}%')
                    
    if epoch % 10 == 0:
        print('Testing')
        val_acc = val()
        if val_acc > best_val:
            best_model_save_path = os.path.join(args.save_dir, f'model_best_{input_features}.pth')
            print(f' ==> Saving best model checkpoint to {best_model_save_path}')
            torch.save(model.state_dict(), best_model_save_path)
            # Save best model on comet
            logger.log_model(f'model_{epoch}_{input_features}.pth', best_model_save_path)
            
        print(f'Epoch {epoch} - Test overall: {100*val_acc:.2f}%')
        model_save_path = os.path.join(args.save_dir, f'model_{epoch}_{input_features}.pth')
        print(f' ==> Saving last model to {model_save_path}')
        torch.save(model.state_dict(), model_save_path)
        

                    
# Validate
val_acc = val()
print("Overall val accuracy: {:06.3f}%".format(100*val_acc))
