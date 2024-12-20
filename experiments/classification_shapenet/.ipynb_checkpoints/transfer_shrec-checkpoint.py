"""
Finetunes the backbone trained on Shapenet, on the SHREC11 dataset
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
sys.path.append(os.path.join(os.path.dirname(__file__), '../classification_shrec11'))
from shrec11_dataset import Shrec11MeshDataset_Simplified, Shrec11MeshDataset_Original


# === Options

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
parser.add_argument("--diffusion_blocks", type=int, help="Number of diffusion blocks", default=8)
parser.add_argument("--diffusion_features", type=int, help="Number of diffusion features", default=64)
parser.add_argument("--n_class", type=int, help="Number of classes to predict", default=16)
parser.add_argument("--k_eig", type=int, help="Size of eigendecomposition", default=128)
parser.add_argument("--device", type=int, help="gpu to use", default=0)
parser.add_argument("--lr", type=float, help="checkpoint to use", default=1e-5)
parser.add_argument("--split_size", type=int, help="How large of a training set per-class default: 10", default=10)
parser.add_argument("--epochs", type=int, help="checkpoint to use", default=200)
parser.add_argument("--dataset_type", type=str, help="which variant of the dataset to use ('original', or 'simplified') default: original", default = 'original')
parser.add_argument("--ckpt_path", type=str, help="checkpoint to use", default='saved_models/model_ckpt.pth')
parser.add_argument("--save_dir", type=str, help="checkpoint to use", default='saved_models_transfer')
args = parser.parse_args()

# system things
device = torch.device(f'cuda:{args.device}')
dtype = torch.float32
n_class = args.n_class
input_features = args.input_features # one of ['xyz', 'hks']
k_eig = args.k_eig

# training settings
n_epoch = args.epochs
lr = args.lr
decay_every = 50
decay_rate = 0.5
augment_random_rotate = (input_features == 'xyz')
label_smoothing_fac = 0.2


# Important paths
base_path = os.path.join(os.path.dirname(__file__), '..', 'classification_shrec11')
op_cache_dir = os.path.join(base_path, 'data', 'op_cache')

if args.dataset_type == "simplified":
    dataset_path = os.path.join(base_path, 'data', 'simplified')
elif args.dataset_type == "original":
    dataset_path = os.path.join(base_path, 'data', 'original')
else:
    raise ValueError("Unrecognized dataset type")


# === Load datasets

# Train dataset
if args.dataset_type == "simplified":
    train_dataset = Shrec11MeshDataset_Simplified(dataset_path, split_size=args.split_size,
                                                  k_eig=k_eig, op_cache_dir=op_cache_dir)
elif args.dataset_type == "original":
    train_dataset = Shrec11MeshDataset_Original(dataset_path, split_size=args.split_size,
                                                  k_eig=k_eig, op_cache_dir=op_cache_dir)
train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

# Test dataset
if args.dataset_type == "simplified":
    test_dataset = Shrec11MeshDataset_Simplified(dataset_path, split_size=None,
                                                 k_eig=k_eig, op_cache_dir=op_cache_dir,
                                                 exclude_dict=train_dataset.entries)
elif args.dataset_type == "original":
    test_dataset = Shrec11MeshDataset_Original(dataset_path, split_size=None,
                                                 k_eig=k_eig, op_cache_dir=op_cache_dir,
                                                 exclude_dict=train_dataset.entries)
test_loader = DataLoader(test_dataset, batch_size=None)





# === Create the model
C_in={'xyz':3, 'hks':16}[input_features] # dimension of input features
C_width=args.diffusion_features


model = diffusion_net.layers.DiffusionNet(C_in=C_in,
                                           C_out=n_class,
                                           C_width=C_width, 
                                           N_block=args.diffusion_blocks, 
                                           last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                                           outputs_at='global_mean', 
                                           dropout=True)
model = model.to(device)


# Load checkpoint
ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt)


# Freeze gradients
for param in model.parameters():
    param.requires_grad = False

# Replace last linear layer and the two last diffusion blocks (assuming there are 8)
model.last_lin = nn.Linear(C_width, n_class).to(device)
model.block_6 = diffusion_net.layers.DiffusionNetBlock(C_width=C_width, dropout=True, mlp_hidden_dims=[C_width,C_width]).to(device)
model.block_7 = diffusion_net.layers.DiffusionNetBlock(C_width=C_width, dropout=True, mlp_hidden_dims=[C_width,C_width]).to(device)


# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train_epoch(epoch):

    # Implement lr decay
    if epoch > 0 and epoch % decay_every == 0:
        global lr 
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

    # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()
    
    correct = 0
    total_num = 0
    for data in tqdm(train_loader):

        # Get data
        verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

        # Move to device
        verts = verts.to(device)
        faces = faces.to(device)
        frames = frames.to(device)
        mass = mass.to(device)
        L = L.to(device)
        evals = evals.to(device)
        evecs = evecs.to(device)
        gradX = gradX.to(device)
        gradY = gradY.to(device)
        labels = labels.to(device)
        
        # Randomly rotate positions
        if augment_random_rotate:
            verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = verts
        elif input_features == 'hks':
            features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)
        features = features.to(device)
        # Apply the model
        preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

        # Evaluate loss
        loss = diffusion_net.utils.label_smoothing_log_loss(preds, labels, label_smoothing_fac)
        loss.backward()
        
        # track accuracy
        pred_labels = torch.max(preds, dim=-1).indices
        this_correct = pred_labels.eq(labels).sum().item()
        correct += this_correct
        total_num += 1

        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()

    train_acc = correct / total_num
    return train_acc


# Do an evaluation pass on the test dataset 
def test():
    
    model.eval()
    
    correct = 0
    total_num = 0
    with torch.no_grad():
    
        for data in tqdm(test_loader):

            # Get data
            verts, faces, frames, mass, L, evals, evecs, gradX, gradY, labels = data

            # Move to device
            verts = verts.to(device)
            faces = faces.to(device)
            frames = frames.to(device)
            mass = mass.to(device)
            L = L.to(device)
            evals = evals.to(device)
            evecs = evecs.to(device)
            gradX = gradX.to(device)
            gradY = gradY.to(device)
            labels = labels.to(device)
            
            # Construct features
            if input_features == 'xyz':
                features = verts
            elif input_features == 'hks':
                features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model
            preds = model(features, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)

            # track accuracy
            pred_labels = torch.max(preds, dim=-1).indices
            this_correct = pred_labels.eq(labels).sum().item()
            correct += this_correct
            total_num += 1

    test_acc = correct / total_num
    return test_acc 


print("Training...")
for epoch in range(n_epoch):
    train_acc = train_epoch(epoch)
    test_acc = test()
    print("Epoch {} - Train overall: {:06.3f}%  Test overall: {:06.3f}%".format(epoch, 100*train_acc, 100*test_acc))
    if epoch % 10 == 0:
        print('Testing')
        test_acc = test()
        print(f'Epoch {epoch} - Test overall: {100*test_acc:.2f}%')
        model_save_path = os.path.join(args.save_dir, f'model_{epoch}_{input_features}_transfer.pth')
        print(f' ==> Saving last model to {model_save_path}')
        torch.save(model.state_dict(), model_save_path)

# Test
test_acc = test()
print("Overall test accuracy: {:06.3f}%".format(100*test_acc))
