from .dataset import ImageDataset, FeatureDataset
from .model import load_image_model, load_text_model, load_transform_model, normalize
import pickle
import torch
import pandas as pd

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
    features = torch.load(feature_file).detach().to(device)
    names_series = pd.Series(pd.read_csv(index_file, header=None, index_col=0))
    return features, names_series


def euclidean_dist(ref_feature, features):
    # Calculate Euclidean distance
    return torch.norm(ref_feature - features, p=2, dim=1)


def cosine_dist(ref_feature, features):
    # Calculate cosine distance
    return 1 - torch.nn.CosineSimilarity()(ref_feature.unsqueeze(0), features)


def k_nearest_neighbors(ref_feature, features, k=10, dist_fn=cosine_dist):
    # return distances and indices of top k nearest neighbors
    dists = dist_fn(ref_feature, features)
    return torch.topk(dists, k,  largest=False)


def get_images_from_caption(caption, image_features, image_names, text_model, text_tokenizer, text_encoder, device, dist_func=cosine_dist, k=50):
    '''
    Return distances and indices of k nearest images of caption
    '''
    # Convert to token
    input_ids = torch.tensor(
        [text_tokenizer.encode(caption, add_special_tokens=True)]).to(device)
    # Use bert-like model to encode (and normalize)
    text_feature = normalize(text_model(input_ids)[0].mean(1))
    # Transform to common space
    text_feature = text_encoder(text_feature)
    # Normalize feature
    text_feature = normalize(text_feature).squeeze()
    # Get top k images
    dists, indices = k_nearest_neighbors(text_feature, image_features, dist_fn=cosine_dist, k=k)
    # Get image filenames from indices
    indices = indices.to('cpu').numpy()
    filenames = image_names.iloc[indices].tolist()
    return dists, filenames
