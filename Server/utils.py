from .dataset import ImageDataset, FeatureDataset
from .model import load_image_model, load_text_model, load_transform_model, normalize, l2norm
import pickle
import torch
import pandas as pd
import os

def convert_image_to_feature(feature_model, transfrom_model, image_folder, save_folder, opt):
    pass


def load_feature(feature_folder, opt):
    pass


def load_all_feature(feature_file, index_file, device):
    '''
    Load all image features and filenames
    Input:
        - feature_file (str): path to features tensor
        - index_file (str): path to filenames pandas series (easier to retrie)
    '''
    features = torch.load(feature_file,map_location=torch.device('cpu')).detach().to(device)
    names_series = pd.Series(pd.read_csv(index_file, index_col=0).iloc[:,0])
    return features, names_series


def euclidean_dist(ref_feature, features):
    # Calculate Euclidean distance
    return torch.norm(ref_feature - features, p=2, dim=1)


def cosine_dist(ref_feature, features):
    # Calculate cosine distance
    return 1 - torch.nn.CosineSimilarity()(ref_feature, features)


def k_nearest_neighbors(ref_feature, features, k=10, dist_fn=cosine_dist):
    # return distances and indices of top k nearest neighbors
    dists = dist_fn(ref_feature, features)
    return torch.topk(dists, k,  largest=False)

stored_sorted = dict()
image_names = dict()

def get_images_from_caption(caption, dataset, image_features_folder, image_names_folder, text_model, text_tokenizer, text_encoder, device, dist_func=cosine_dist, k=50,start_from=0):
    '''
    Return distances and indices of k nearest images of caption
    '''
    if image_features_folder+image_names_folder not in image_names:
        names_series = []
        for feature_file in os.listdir(image_features_folder):
            name_file = os.path.join(image_names_folder, os.path.splitext(feature_file)[0] + '.csv')
            filenames = pd.Series(pd.read_csv(name_file,header=None, index_col=0).iloc[:,0])
            names_series.append(filenames)
        image_names[dataset] = pd.concat(names_series, ignore_index=True)

    if start_from==0:
        print('Calculating for caption: ',caption)
        # Convert to token
        input_ids = torch.tensor(
            [text_tokenizer.encode(caption, add_special_tokens=True)]).to(device)
        # Use bert-like model to encode (and normalize)
        text_feature = l2norm(text_model(input_ids)[0][-1][0,...].unsqueeze(0))
        
        # Transform to common space
        text_feature = text_encoder(text_feature)
        # Normalize feature
        # text_feature = normalize(text_feature).squeeze()

        # Iterate throgh all features
        dists = []
        for feature_file in os.listdir(image_features_folder):
            feature_file = os.path.join(image_features_folder, feature_file)
            image_features = torch.load(feature_file,map_location=device).detach().to(device)
            dists.append(dist_func(text_feature, image_features))
        dists = torch.cat(dists, dim=0)
        # Get top k images
        #dists, indices = k_nearest_neighbors(text_feature, image_features, dist_fn=dist_func, k=k)
        stored_sorted['dists'], stored_sorted['indices'] = torch.topk(dists,100,largest=False)
    
    indices = stored_sorted['indices'][start_from:start_from+k]
    dists = stored_sorted['dists'][start_from:start_from+k]

    # Get image filenames from indices
    indices = indices.to('cpu').numpy()
    filenames = image_names[dataset].iloc[indices].tolist()
    return dists, filenames
