import time
import numpy
from tqdm import tqdm
import random
import os

import sys

from scipy.spatial.distance import cdist
import scipy.sparse as spa
from fastdist import fastdist
import numpy as np
import osqp
from qpsolvers import solve_qp
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

import torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

from .sampler import SubsetSequentialSampler
from utils import get_average_node_per_molecule



def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('ConcreteDropout'):
            m.train()

def random_sel(labeled_indices, unlabeled_indices, k=1500):
    return random.sample(unlabeled_indices, k) + labeled_indices


def mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=False):

    time_start = time.time()
    pred_matrix = torch.tensor([]).to(device)

    for i in range(20):
        unlabeled_loader = DataLoader(data, batch_size=64, 
                            sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
                            pin_memory=True, drop_last=False)

        print('Len_unlabloader: ', len(unlabeled_loader))
        
        #set model to evaluation mode
        model.eval()
        enable_dropout(model)

        with torch.cuda.device(device):
            preds = torch.tensor([]).to(device)

        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(unlabeled_loader)):
                batch_data = batch_data.to(device)
            
                with torch.no_grad():
                    _, _, out, _ = model(batch_data)
                    # print(out.shape)
                preds = torch.cat([preds, out.detach_()], dim=0)

        pred_matrix = torch.cat([pred_matrix, preds], dim=1)
        # print(pred_matrix[:10, :])
    
    
    variance = torch.var(pred_matrix, dim=1)
    if with_diversity:
        return variance

    arg = torch.argsort(variance, descending=True).cpu()
    return torch.tensor(unlabeled_indices)[arg[:k]], (time.time()-time_start)/60

##uncertainty rep using euclidean distance
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def compute_USR(mol_pos):
    # Function to calculate four moments for a given set of values
    def calculate_moments(values):
        mean_value = values.mean()
        variance = values.var()
        skewness = (torch.abs((values - mean_value) ** 3)).mean().sqrt()
        kurtosis = ((values - mean_value) ** 4).mean().sqrt()
        return [mean_value, variance, skewness, kurtosis]

    # Function to calculate the angle between two vectors
    def calculate_angle(a, b):
        cos_angle = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        return angle

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

    # Compute angles with the reference vector for atoms to centroid
    angles1 = [calculate_angle(reference_vector, pt - centroid) for pt in mol_pos]
    angles11 = [calculate_angle(reference_vector1, pt - centroid) for pt in mol_pos]
    angles12 = [calculate_angle(reference_vector2, pt - centroid) for pt in mol_pos]
    
    # Compute angles with the reference vector for all possible atom pairs
    angles2 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector, mol_pos[j] - mol_pos[i])
                angles2.append(angle)
    angles21 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector1, mol_pos[j] - mol_pos[i])
                angles21.append(angle)
    angles22 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector2, mol_pos[j] - mol_pos[i])
                angles22.append(angle)

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


def pairwise_usr_similarity(matrix1, matrix2=None):
    if matrix2 is None:
        matrix2 = matrix1
        
    # Compute distances
    distances = cdist(matrix1.cpu().numpy(), matrix2.cpu().numpy(), 'cityblock') / 4.0 + 1.0
    similarities = 1.0 / distances
    
    # Convert similarities to a torch tensor
    tensor_similarities = torch.tensor(similarities, dtype=torch.float32)
    return tensor_similarities




def compute_uncertainty_diversity(model, labeled_indices, unlabeled_indices, data, device, k=1500,expt=1, selection_method='unc_div'):

    time_start = time.time()
    unlabeled_loader = DataLoader(data, batch_size=64, 
                        sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
                        pin_memory=True, drop_last=False)
   
    print('Len_unlabloader: ', len(unlabeled_loader))
    
    #set model to evaluation mode
    model.eval()

    with torch.cuda.device(device):
        #features = torch.tensor([]).to(device)
        features1 = torch.tensor([]).to(device)

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
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
    
    similarity_matrix = similarity_matrix.cpu().numpy()
    print('similarity_matrix shape: ', similarity_matrix.shape)


    uncertainty = mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=True)
    uncertainty = uncertainty/uncertainty.max()
    uncertainty = uncertainty.cpu().numpy()

    ### setting upper bound lower bounds and constraints
    lb = np.zeros(len(unlabeled_indices))
    ub = np.ones(len(unlabeled_indices))

    A = np.ones(len(unlabeled_indices))
    b = 1.0 * np.array([k])
    
    # un_rep = uncertainty 

    #using osqp
    print(f'Solving qp....')
#     prob = osqp.OSQP()

# # Setup workspace and change alpha parameter
#     prob.setup(diversity_matrix, uncertainty, A, lb, ub, alpha=1.0, warm_starting=True)
#     res = prob.solve()
    res = solve_qp(P=similarity_matrix, q=uncertainty, G=None, A=A, b=b, lb=lb, ub=ub, solver='osqp', verbose=True)
##
# Solve problem 
    
    arg = np.argsort(res)
    query_indices = np.array(unlabeled_indices)[arg[-k:]]
    return labeled_indices+query_indices.tolist(), (time.time()-time_start)//60



def compute_uncertainty_diversity_gpu(model, labeled_indices, unlabeled_indices, data, device, k=1500,expt=1, selection_method='unc_div'):

    time_start = time.time()
    #set model to evaluation mode
    model.eval()

     #load similarity matrix 
    similarity_matrix = torch.load(f'runs/{selection_method}/run{expt}/tensor.pt')
    similarity_matrix.cpu().numpy()
    
    uncertainty = mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=True)
    uncertainty = uncertainty/uncertainty.max()
    uncertainty = uncertainty.cpu().numpy()

    ### setting upper bound lower bounds and constraints
    lb = np.zeros(len(unlabeled_indices))
    ub = np.ones(len(unlabeled_indices))

    A = np.ones(len(unlabeled_indices))
    b = 1.0 * np.array([k])
    
    # un_rep = uncertainty 

    #using osqp
    print(f'Solving qp....')
    P = spa.csc_matrix(similarity_matrix)
    q = uncertainty
    G = None 
    h = None


    A_osqp = None
    l_osqp = None
    u_osqp = None
    if G is not None and h is not None:
        A_osqp = G
        l_osqp = np.full(h.shape, -np.infty)
        u_osqp = h
    if A is not None and b is not None:
        A_osqp = A if A_osqp is None else spa.vstack([A_osqp, A], format="csc")
        l_osqp = b if l_osqp is None else np.hstack([l_osqp, b])
        u_osqp = b if u_osqp is None else np.hstack([u_osqp, b])
    if lb is not None or ub is not None:
        lb = lb if lb is not None else np.full(q.shape, -np.infty)
        ub = ub if ub is not None else np.full(q.shape, +np.infty)
        E = spa.eye(q.shape[0])
        A_osqp = E if A_osqp is None else spa.vstack([A_osqp, E], format="csc")
        l_osqp = lb if l_osqp is None else np.hstack([l_osqp, lb])
        u_osqp = ub if u_osqp is None else np.hstack([u_osqp, ub])


    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A_osqp, l=l_osqp, u=u_osqp, alpha=1.0, warm_starting=True)
    res = solver.solve()
    # print(res)
    res = res.x
    # res = solve_qp(P=diversity_matrix, q=uncertainty, G=None, A=A, b=b, lb=lb, ub=ub, solver='osqp', verbose=True)
    qp_time = time.time() - qp_time
    
    arg = np.argsort(res)
    
    selected_indices = arg[-k:] ###gives index
    
    ''' ########### Here ############ '''
    unselected_indices = torch.tensor(np.setdiff1d(np.arange(similarity_matrix.shape[0]), selected_indices))
    
    #choose subset of diversity matrix containing unselected indices
    
    similarity_matrix = similarity_matrix[:,unselected_indices]
    similarity_matrix = similarity_matrix[unselected_indices,:]
    
    torch.save(similarity_matrix, f'runs/{selection_method}/run{expt}/tensor.pt')
    
    
    query_indices = np.array(unlabeled_indices)[arg[-k:]]
    return labeled_indices+query_indices.tolist(), (time.time()-time_start)//60

def compute_uncertainty_diversity_fast(model, labeled_indices, unlabeled_indices, data, device, k=1500,expt=1, selection_method='unc_div'):

    time_start = time.time()
    #set model to evaluation mode
    model.eval()

     #load similarity matrix 
    similarity_matrix = torch.load(f'runs/{selection_method}/run{expt}/tensor.pt')
    
    print('similarity_matrix shape: ', similarity_matrix.shape)


    uncertainty = mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=True)
    uncertainty = uncertainty/uncertainty.max()
    uncertainty = uncertainty.cpu().numpy()

    ### setting upper bound lower bounds and constraints
    lb = np.zeros(len(unlabeled_indices))
    ub = np.ones(len(unlabeled_indices))

    A = np.ones(len(unlabeled_indices))
    b = 1.0 * np.array([k])
    
    # un_rep = uncertainty 

    #using osqp
    print(f'Solving qp....')

    res = solve_qp(P=similarity_matrix, q=uncertainty, G=None, A=A, b=b, lb=lb, ub=ub, solver='osqp', verbose=True)

# Solve problem 
    
    arg = np.argsort(res)
    
    selected_indices = arg[-k:] ###gives index
    
    ''' ########### Here ############ '''
    unselected_indices = torch.tensor(np.setdiff1d(np.arange(similarity_matrix.shape[0]), selected_indices))
    
    #choose subset of diversity matrix containing unselected indices
    
    similarity_matrix = similarity_matrix[:,unselected_indices]
    similarity_matrix = similarity_matrix[unselected_indices,:]
    
    torch.save(similarity_matrix, f'runs/{selection_method}/run{expt}/tensor.pt')
    
    
    query_indices = np.array(unlabeled_indices)[arg[-k:]]
    return labeled_indices+query_indices.tolist(), (time.time()-time_start)//60


def get_loss(model, lossnet, unlabeled_loader, device):
    model.eval()
    lossnet.eval()
    with torch.cuda.device(device):
        loss = torch.tensor([]).to(device)

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
            batch_data = batch_data.to(device)
            with torch.no_grad():
                _, feature_dict, _ = model(batch_data)
                node_features = feature_dict['node_features']

            features_list = []
            for n_feature in node_features:
                av_node_features = get_average_node_per_molecule(batch_data, n_feature, device)
                features_list.append(av_node_features)
           
            lossnet_out = lossnet(features_list)                
            lossnet_out = lossnet_out.view(lossnet_out.size(0))
            loss = torch.cat((loss, lossnet_out), 0)
    
    return torch.abs(loss)


def lloss_sel(model, lossnet, unlabeled_indices, data, device, k=1500):
    time_start = time.time()
    unlabeled_loader = DataLoader(data, batch_size=64, 
                                    sampler=SubsetSequentialSampler(unlabeled_indices), 
                                    pin_memory=True)

    loss = get_loss(model, lossnet, unlabeled_loader, device)
    arg = torch.argsort(loss, descending=True).cpu()
    print(arg)
    print('Querying lloss   ----- Len : ', len(arg))
    print('len data: ', len(unlabeled_indices))
    return torch.tensor(unlabeled_indices)[arg[:k]], (time.time()-time_start)/60


##node features
def coreset(model, labeled_indices, unlabeled_indices, data, device, k=1500):

    time_start = time.time()
    unlabeled_loader = DataLoader(data, batch_size=8, 
                        sampler=SubsetSequentialSampler(labeled_indices+unlabeled_indices), # more convenient if we maintain the order of subset
                        pin_memory=True, drop_last=False)

    print('Len_unlabloader: ', len(unlabeled_loader))
    
    #set model to evaluation mode
    model.eval()

    with torch.cuda.device(device):
        features = torch.tensor([]).to(device)

    with torch.no_grad():
        i=0
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
            batch_data = batch_data.to(device)
          
            with torch.no_grad():
                _, feature_dict, _= model(batch_data)
                node_features = feature_dict['node_features']

            av_node_features = get_average_node_per_molecule(batch_data, node_features[-1], device)
            features = torch.cat((features, av_node_features), 0)
        
        
    feat = features.detach().cpu().numpy()
    print(feat.shape)

    train_predictions = feat[:len(labeled_indices)]
    predictions = feat[len(labeled_indices):]

    print(f'Train_shape: {train_predictions.shape}:::: un_shape: {predictions.shape}')
    '''
    This functions takes feature values of labeled dataset and unlabeled dataset and return unlabeled indices based on the coreset method.
    '''
    
    
    subset_indices = numpy.array(unlabeled_indices)

    query_indices = []
    unlabeled_pairwise_distance = fastdist.matrix_pairwise_distance(predictions, metric=fastdist.euclidean, metric_name="euclidean", return_matrix=True)
    distance = cdist(predictions, train_predictions, metric="euclidean")
    # unlabeled_pairwise_distance = fastdist.matrix_pairwise_distance(predictions[subset_indices], metric=fastdist.euclidean, metric_name="euclidean", return_matrix=True)
    # distance = cdist(predictions[subset_indices], train_predictions, metric="euclidean")
    print(f'Samples to select {k}')
    for i in range(0, k):
        t1 = time.time()
        min_distances = numpy.min(distance, axis=1)
        max_idx = numpy.argmax(min_distances)
        print(f'selecting {i}th sample of index {subset_indices[max_idx]}')
        if subset_indices[max_idx] in labeled_indices:
            print('Already in labeled;;;;stop stop')
            break
        query_indices.append(subset_indices[max_idx])
        # print(distance.shape, " ", unlabeled_pairwise_distance[max_idx].shape, end = "\r")
        distance = numpy.append(distance, numpy.array([unlabeled_pairwise_distance[max_idx]]).T, 1)
        distance = numpy.delete(distance, max_idx, 0)
        unlabeled_pairwise_distance = numpy.delete(unlabeled_pairwise_distance, max_idx, 0)
        unlabeled_pairwise_distance = numpy.delete(unlabeled_pairwise_distance, max_idx, 1)
        subset_indices = numpy.delete(subset_indices, max_idx) 
        print(f'Time taken for selecting {i+1}th sample: {(time.time() - t1)} secs\n') 
    
    return labeled_indices+query_indices, (time.time()-time_start)//60
