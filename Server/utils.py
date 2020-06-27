from .dataset import ImageDataset, FeatureDataset
from .model import load_text_model, load_transform_model, normalize, l2norm
import pickle
import torch
import pandas as pd
import os
import datetime
import time

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

def get_images_from_caption(caption, image_features_folder, image_names, text_model, text_tokenizer, 
                            text_encoder, device, max_seq_len = 64, dist_func=cosine_dist, k=50,start_from=0):
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

def get_images_from_caption_subset(caption, subset, image_features_folder, image_names, reversed_names, text_model, 
                                    text_tokenizer, text_encoder, device, max_seq_len = 64, dist_func=cosine_dist, k=50,start_from=0):
    tokenizer_res = text_tokenizer.encode_plus(caption, add_special_tokens=True, pad_to_max_length=True, max_length=max_seq_len, return_attention_mask=True, return_token_type_ids=False)
    input_ids = torch.tensor([tokenizer_res['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenizer_res['attention_mask']]).to(device)

    text_feature = text_model(input_ids, attention_mask=attention_mask)
    text_feature = l2norm(text_feature)
    text_feature = text_encoder(text_feature)
    
    # Prepare filename dict
    filename_dict = dict()
    for filename in subset:
        subfolder = filename.split('/')[0]
        if subfolder not in filename_dict:
            filename_dict[subfolder] = []
        filename_dict[subfolder].append(filename)
    
    # Calculate
    dists = []
    sub_image_names = []
    for subfolder in filename_dict:
        feature_file = os.path.join(image_features_folder, subfolder + '.pth')
        image_features = torch.load(feature_file,map_location=device).detach().to(device)
        image_features = image_features[reversed_names[filename_dict[subfolder]].values]
        dists.append(dist_func(text_feature, image_features))
        sub_image_names+=filename_dict[subfolder]
    dists = torch.cat(dists, dim=0)
    sub_image_names = pd.Series(sub_image_names)
    # Get top k images
    #dists, indices = k_nearest_neighbors(text_feature, image_features, dist_fn=dist_func, k=k)
    dists_sorted, indices_sorted = torch.topk(dists,k+start_from,largest=False)
    
    indices = indices_sorted[start_from:k+start_from]
    dists = dists_sorted[start_from:k+start_from]
    indices = indices.to('cpu').numpy()
    filenames = sub_image_names.iloc[indices].tolist()
    return dists, filenames

def get_image_set_before_time(concepts, minute_id, minute_before):
    res = set()
    time = datetime.datetime(int(minute_id[:4]), int(minute_id[4:6]), int(minute_id[6:8]), int(minute_id[9:11]), int(minute_id[11:]))
    time_before = time - datetime.timedelta(minutes=minute_before)
    minute_id_before = time_before.strftime("%Y%m%d_%H%M")
    res = set(concepts.loc[(concepts['minute_id'] >= minute_id_before) & (concepts['minute_id'] <= minute_id)]['image_path'])
    return res

def get_similar_images(image_path, similar_feature_folder, similar_filename_folder, device, k=50, start_from=0):
    # Read filename series
    names_series = []
    reversed_names = []
    for feature_file in os.listdir(similar_feature_folder):
        name_file = os.path.join(similar_filename_folder, os.path.splitext(feature_file)[0] + '.csv')
        filenames = pd.Series(pd.read_csv(name_file,header=None, index_col=0).iloc[:,0])
        names_series.append(filenames)
        reversed_names.append(pd.Series(filenames.index.values, index=filenames))
    image_names = pd.concat(names_series, ignore_index=True)
    reversed_names = pd.concat(reversed_names)

    subfolder = image_path.split('/')[0]
    path = os.path.join(similar_feature_folder, subfolder + '.pth')
    features = torch.load(path,map_location=device).detach().to(device)
    ref_feature = features[reversed_names[image_path]].unsqueeze(0)
    # Calculate
    dists = []
    for feature_file in os.listdir(similar_feature_folder):
        feature_file = os.path.join(similar_feature_folder, feature_file)
        image_features = torch.load(feature_file,map_location=device).detach().to(device)
        dists.append(cosine_dist(ref_feature, image_features))
    dists = torch.cat(dists, dim=0)
    dists_sorted, indices_sorted = torch.topk(dists,k + start_from + 1,largest=False)
    
    indices = indices_sorted[start_from + 1:k+start_from + 1]
    dists = dists_sorted[start_from + 1:k+start_from + 1]
    indices = indices.to('cpu').numpy()
    filenames = image_names.iloc[indices].tolist()
    return dists, filenames