import argparse

import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from fastdist import fastdist

from torch_geometric.loader import DataLoader
from datasets import QM93D

from al.sampler import SubsetSequentialSampler


def get_split_idx(train_size, np_file):
    indices = np.arange(train_size)
    labeled_set = np.load(f"{np_file}")
    unlabeled_set = np.setdiff1d(indices, labeled_set)
    
    assert np.intersect1d(labeled_set, unlabeled_set).size == 0 
    assert np.union1d(labeled_set, unlabeled_set).size == train_size
    return {'lab':labeled_set.tolist(), 'unlab': unlabeled_set.tolist()}

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def pairwise_usr_similarity(matrix1, matrix2=None):
    if matrix2 is None:
        matrix2 = matrix1
        
    # Compute distances
    distances = torch.cdist(matrix1, matrix2, p=1) / 4.0 + 1.0
    similarities = 1.0 / distances

    return similarities


  # Function to calculate four moments for a given set of values
def calculate_moments(values):
        mean_value = values.mean()
        variance = values.var()
        skewness = (torch.abs((values - mean_value) ** 3)).mean().sqrt()
        kurtosis = ((values - mean_value) ** 4).mean().sqrt()
        return [mean_value, variance, skewness, kurtosis]

# Function to calculate the angle between two vectors
def calculate_angle(ref, matrix):
  #a = ref vector and b = matrix
  ref = ref/ref.norm()

  matrix = matrix/matrix.norm(dim=1, keepdim=True)

  cosine_similarities = torch.matmul(matrix, ref)

  angles = torch.acos(torch.clamp(cosine_similarities, -1, 1))

  return angles
  
def calculate_diff(mol_pos):
  #a = ref vector and b = matrix
  
    diffs =  mol_pos.unsqueeze(0) - mol_pos.unsqueeze(1)  # Shape: (n, n, 3)
    
  # Mask out the diagonal elements (i.e., self-differences)
    mask = torch.eye(len(mol_pos), dtype=torch.bool)
    
    diffs = diffs[~mask].view(len(mol_pos), -1, mol_pos.size(1))
    
    # print(diffs.shape)
    diffs  = diffs.view(-1,3)
    return diffs

def calculate_angle_old(a, b):
 
    cos_angle = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    angle = torch.acos(torch.clamp(cos_angle, -1, 1))
    return angle

def compute_USR(mol_pos):
    # print(len(mol_pos), type(mol_pos))
    
    # Calculate the centroid
    centroid = mol_pos.mean(dim=0)

    # Find the first farthest point from the centroid
    distances_to_centroid = torch.norm(mol_pos - centroid, dim=1)
    farthest_point = mol_pos[torch.argmax(distances_to_centroid)]
    closest_point = mol_pos[torch.argmin(distances_to_centroid)]
    
    distances_from_farthest = torch.norm(mol_pos - farthest_point, dim=1)
    farfromfarthest_point = mol_pos[torch.argmax(distances_from_farthest)]
    
    # Calculate the reference vector from centroid to the farthest point
    reference_vector = farthest_point - centroid
    reference_vector1 = closest_point - centroid
    reference_vector2 = farfromfarthest_point - centroid


    '''rs'''
    # Compute angles with the reference vector for atoms to centroid
    pt_centroid = mol_pos - centroid
    

    angles1 = calculate_angle(reference_vector, pt_centroid)
    angles1 = list(torch.split(angles1, 1))
    
    angles11 = calculate_angle(reference_vector1, pt_centroid)
    angles11 = list(torch.split(angles11, 1))
    
    angles12 = calculate_angle(reference_vector2, pt_centroid)
    angles12 = list(torch.split(angles12, 1))
    
    '''rs'''
    #######
    '''rs'''
    # Compute angles with the reference vector for all possible atom pairs
    diffs = calculate_diff(mol_pos)
    
    angles2 = calculate_angle(reference_vector, diffs)
    angles2 = list(torch.split(angles2, 1))
    
    angles21 = calculate_angle(reference_vector1, diffs)
    angles21 = list(torch.split(angles21, 1))
    
    angles22 = calculate_angle(reference_vector2, diffs)
    angles22 = list(torch.split(angles22, 1))
    
    '''rs'''

  
    # Calculate moments for distances from 4 reference points
    distances_to_point1 = torch.norm(mol_pos - farthest_point, dim=1)
    farthest_point2 = mol_pos[torch.argmax(distances_to_point1)]
    
    distances_to_point2 = torch.norm(mol_pos - centroid, dim=1)
    farthest_point3 = mol_pos[torch.argmin(distances_to_point2)]

    points = [centroid, farthest_point, farthest_point2, farthest_point3]
    moments_all = []
    for point in points:
        distances = torch.norm(mol_pos - point, dim=1)
        moments_all.extend(calculate_moments(distances))
    #print("calculate_moments(distances):", torch.tensor(calculate_moments(distances)))

    # Add moments for both sets of angles
    moments_all.extend(calculate_moments(torch.tensor(angles1)))
    moments_all.extend(calculate_moments(torch.tensor(angles11)))
    moments_all.extend(calculate_moments(torch.tensor(angles12)))
    moments_all.extend(calculate_moments(torch.tensor(angles2)))
    moments_all.extend(calculate_moments(torch.tensor(angles21)))
    moments_all.extend(calculate_moments(torch.tensor(angles22)))
    #print("Shape of moments_all:", torch.tensor(moments_all).shape)

    return torch.tensor(moments_all)


def similarity_our(opt):
    device = opt.device
    batch_size = opt.batch_size
    dataset = opt.dataset
    selection_method = opt.selection_method
    init_size = opt.init_size
    train_size = opt.train_size
    valid_size = opt.valid_size
    
    
    dataset = QM93D(root='dataset/', processed_fn=dataset)
    '''
    Put original init set in ## runs/{method}/run{expt}/init_set.npy
    '''
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=train_size,valid_size=valid_size, 
                                      init_size=init_size, seed=42, method= selection_method)

    train_dataset, valid_dataset, test_dataset, unlab_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]

    unlab_loader =DataLoader(dataset, batch_size=1, 
                            sampler=SubsetSequentialSampler(split_idx['unlabeled']), # more convenient if we maintain the order of subset
                            pin_memory=True, drop_last=False)
    print(unlab_loader)

    with torch.cuda.device(device):
        features1 = torch.tensor([]).to(device)

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(unlab_loader)):
            batch_data = batch_data.to(device)
          
            with torch.no_grad():
                z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
            usr_features = []
            for mol_idx in batch.unique():
            
                mol_pos = pos[batch == mol_idx]
                usr_features.append(compute_USR(mol_pos))
            usr_features = torch.stack(usr_features).to(device)
            features1 = torch.cat((features1, usr_features), 0)

    unlab_predictions = features1
    

    similarity_matrix = pairwise_usr_similarity(unlab_predictions)
    similarity_matrix = similarity_matrix/similarity_matrix.max()
    similarity_matrix.fill_diagonal_(30.0)
    
    return similarity_matrix.cpu()
