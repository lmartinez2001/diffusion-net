import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import kagglehub
import json
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the 
import diffusion_net


class ShapenetDataset(Dataset):
    def __init__(self, root=None, split='train', op_cache_dir='op_cache', k_eig=128):
        self.split = split
        self.root = root if root else self._get_root()
        self.split_list = self._get_split_list()
        
        self.op_cache_dir = op_cache_dir
        self.k_eig = k_eig

        self.verts_list = []
        self.labels_list = []
        self.files_list = []
        
        print('Loading dataset...')
        for label, _, sample_rel_path in tqdm(self.split_list):
            model_path = os.path.join(self.root, sample_rel_path)
            verts = np.load(model_path)
            verts = torch.tensor(verts).float()
            verts = diffusion_net.geometry.normalize_positions(verts)
            self.verts_list.append(verts)
            self.labels_list.append(torch.tensor(label))
            self.files_list.append(model_path)
            
            
    def _get_root(self):
        print('root not specified, automatically retrieving data...')
        root = kagglehub.dataset_download("jeremy26/shapenet-core")
        print(f'dataset stored in {root}')
        dataset_name = 'Shapenetcore_benchmark'
        return os.path.join(root, dataset_name)

    
    def _get_split_list(self):
        def read_split(path):
            with open(path, 'r') as f:
                split = json.load(f)
            return split
        
        split_path = os.path.join(self.root, f'{self.split}_split.json')
        split_list = read_split(split_path)
        
        return split_list
    
    
    def __len__(self):
        return len(self.verts_list)

    
    def __getitem__(self, idx):
        path = self.files_list[idx]
        verts = self.verts_list[idx]
        label = self.labels_list[idx]
        path = self.files_list[idx]
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(verts, 
                                                                                           torch.tensor([]), 
                                                                                           k_eig=self.k_eig, 
                                                                                           op_cache_dir=self.op_cache_dir
                                                                                          )
        return path, verts, frames, mass, L, evals, evecs, gradX, gradY, label
    
    
    
# TEST
if __name__ == '__main__':
    train_set = ShapenetDataset(split='train')
    print(len(train_set))
    print(f'path: {train_set[0][0]}')
    print(f'sample shape: {train_set[0][1].shape}')
    print(f'label: {train_set[0][-1]}')
    
    val_set = ShapenetDataset(split='val')
    print(len(val_set))
    print(f'path: {val_set[0][0]}')
    print(f'sample shape: {val_set[0][1].shape}')
    print(f'label: {val_set[0][-1]}')
