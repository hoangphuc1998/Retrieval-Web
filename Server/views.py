from django.shortcuts import render
import torch
from .utils import get_images_from_caption, load_all_feature, load_text_model, load_transform_model, cosine_dist, euclidean_dist
import pickle
import json
import os
from django.http import JsonResponse, HttpResponse
import base64
# Create your views here.

# Global variables
path = {    
    'COCO':{
        'image_feature_folder' : '/home/lehoanganh298/Projects/datasets/COCO/COCO_val_image_transform',
        'filename_folder' : '/home/lehoanganh298/Projects/datasets/COCO/COCO_filename',
        'option_dict_path' : '/home/lehoanganh298/Projects/models/DualEncoding-XEncoder/options.pkl',
        'text_encoder_path' : '/home/lehoanganh298/Projects/models/DualEncoding-XEncoder/DualEncoding_Ver2/best/text_encoder.pth',
        'image_folder' : '/home/lehoanganh298/Projects/datasets/COCO/val2014',
    },
    'LSC': {
        'image_feature_folder' : '/home/lehoanganh298/Projects/datasets/LSC2020/LSC_transform',
        'filename_folder' : '/home/lehoanganh298/Projects/datasets/LSC2020/LSC_filename',
        'option_dict_path' : '/home/lehoanganh298/Projects/models/DualEncoding-XEncoder/options.pkl',
        'text_encoder_path' : '/home/lehoanganh298/Projects/models/DualEncoding-XEncoder/DualEncoding_Ver2/best/text_encoder.pth',
        'image_folder' : '/home/lehoanganh298/Projects/datasets/LSC2020/Volumes/Samsung_T5/DATASETS/LSC2020',
    }
}
text_model = None
text_tokenizer = None
text_encoder = None
opt = None
device = torch.device('cpu')


def home(request):
    '''
    Load all model and features to memory
    '''
    global text_model, text_tokenizer, text_encoder, image_features, image_names
    with open(path['COCO']['option_dict_path'], 'rb') as f:
        opt = pickle.load(f)
    text_model, text_tokenizer = load_text_model(
        opt['text_model_type'], opt['text_model_pretrained'], device)
    text_encoder = load_transform_model(opt, path['COCO']['text_encoder_path'], device)
    # image_features, image_names = load_all_feature(
    #     image_feature_path, filename_path, device)
    return HttpResponse('Setup done!')

def get_images(request, caption, dataset, dist_func, k, start_from):
    global text_model, text_tokenizer, text_encoder, image_features, image_names
    if dist_func == 'cosine':
        dist_func = cosine_dist
    else:
        dist_func = euclidean_dist
    dists, filenames = get_images_from_caption(caption=caption,
                                                dataset=dataset,
                                              image_features_folder=path[dataset]['image_feature_folder'],
                                              image_names_folder=path[dataset]['filename_folder'],
                                              text_model=text_model,
                                              text_tokenizer=text_tokenizer,
                                              text_encoder=text_encoder,
                                              device=device,
                                              dist_func=dist_func,
                                              k=k, start_from=start_from)
    response_data = dict()
    response_data['image'] = []
    #TODO: Add different image dataset
    for filename in filenames:
        with open(os.path.join(path[dataset]['image_folder'], filename), 'rb') as f:
            response_data['image'].append(base64.b64encode(f.read()).decode('utf-8'))
    response_data['dists'] = dists.tolist()
    # print(dists)
    return JsonResponse(response_data)