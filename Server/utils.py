from .dataset import ImageDataset, FeatureDataset
from .model import load_text_model, load_transform_model, normalize, l2norm
import pickle
import torch
import pandas as pd
import os

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

# stored_sorted = dict()
image_names = dict()

def get_images_from_caption(caption, image_features_folder, image_names, text_model, text_tokenizer, text_encoder, device, max_seq_len = 64, dist_func=cosine_dist, k=50,start_from=0):
    '''
    Return distances and indices of k nearest images of caption
    '''
    # Convert to token
    tokenizer_res = text_tokenizer.encode_plus(caption, add_special_tokens=True, pad_to_max_length=True, max_length=max_seq_len, return_attention_mask=True, return_token_type_ids=False)
    input_ids = torch.tensor([tokenizer_res['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenizer_res['attention_mask']]).to(device)
    # Use bert-like model to encode (and normalize)
    text_feature = l2norm(text_model(input_ids=input_ids, attention_mask=attention_mask))
    
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
    dists_sorted, indices_sorted = torch.topk(dists,start_from + k,largest=False)
    
    indices = indices_sorted[start_from:start_from+k]
    dists = dists_sorted[start_from:start_from+k]

    # Get image filenames from indices
    indices = indices.to('cpu').numpy()
    filenames = image_names.iloc[indices].tolist()
    return dists, filenames
