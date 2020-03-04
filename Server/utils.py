from .dataset import ImageDataset, FeatureDataset
from .model import load_image_model, load_text_model, load_transform_model

def convert_image_to_feature(feature_model, transfrom_model, image_folder, save_folder, opt):
    pass

def load_feature(feature_folder, opt):
    pass

def get_images_from_caption(caption, image_features, text_model, text_tokenizer, transform_model, dist_func, k=50):
    pass

