from django.apps import AppConfig
from pathlib import Path
import json
import torch
from .utils import *
import os
import pandas as pd
import glob

def getDOW(date_str):
    return int(datetime.datetime.strptime(date_str, '%Y%m%d').weekday())

class ServerConfig(AppConfig):
    name = 'Server'
    PARENT_PATH = Path('..')
    path = {
        'sajem_feature_folder' : PARENT_PATH/'features/sajem/',
        'resnet_feature_folder': PARENT_PATH/'features/resnet/',
        'filename_folder': PARENT_PATH/'features/filename/',
        'option_dict_path' : PARENT_PATH/'models/options.json',
        'text_encoder_path' :  PARENT_PATH/'models/text_encoder.pth',
        'bert_model_path' : PARENT_PATH/'models/bert_model.pth',
        'metadata_path': PARENT_PATH/'metadata/metadata.csv',
        'concepts_path': PARENT_PATH/'metadata/visual_concepts.csv',
    }
    # Pytorch device
    device = torch.device('cpu')

    # Load json config file
    with open(path['option_dict_path'], 'r') as f:
        opt = json.load(f)

    # Load text model
    text_model, text_tokenizer = load_text_model(
        opt['text_model_type'], opt['text_model_pretrained'], opt['output_bert_model'], device, path['bert_model_path'])
    # Load text transform model
    text_encoder = load_transform_model(opt, path['text_encoder_path'], device)
    # Load SAJEM filenames
    names_series = []
    reversed_names_series = []
    for feature_file in os.listdir(path['sajem_feature_folder']):
        name_file = os.path.join(path['filename_folder'], os.path.splitext(feature_file)[0] + '.csv')
        filenames = pd.Series(pd.read_csv(name_file,header=None, index_col=0).iloc[:,0])
        names_series.append(filenames)
        reversed_names_series.append(pd.Series(filenames.index.values, index=filenames))
    image_names = pd.concat(names_series, ignore_index=True)
    reversed_names_series = pd.concat(reversed_names_series)
    # Query with metadata setup
    metadata = pd.read_csv(path['metadata_path'])
    metadata = metadata[['minute_id', 'semantic_name']]
    metadata = metadata.dropna()
    concepts = pd.read_csv(path['concepts_path'])
    concepts = concepts[['minute_id', 'image_path']]
    concepts['image_path'] = concepts['image_path'].str.slice(17)
    metadata = metadata.merge(concepts)
    # metadata['date'] = metadata['minute_id'].str.slice(0,8)
    # metadata['hour'] = metadata['minute_id'].str.slice(9,11)
    # metadata['minute'] = metadata['minute_id'].str.slice(11)
    concepts['dow'] = concepts.minute_id.str.slice(0,8).apply(getDOW)
    concepts['day'] = concepts.minute_id.str.slice(6,8).astype('int32')
    concepts['month'] = concepts.minute_id.str.slice(4,6).astype('int32')
    concepts['year'] = concepts.minute_id.str.slice(0,4).astype('int32')
    print('Setup done')