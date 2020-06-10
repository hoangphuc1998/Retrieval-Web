from django.shortcuts import render
import torch
from .utils import *
import pickle
import json
import os
import pandas as pd
from django.http import JsonResponse, HttpResponse
import base64
# Create your views here.

# Global variables
path = {
    'image_feature_folder' : '/dataset/LSC/transform/',
    'filename_folder' : '/dataset/LSC/filename/',
    'option_dict_path' : '/options.json',
    'text_encoder_path' : '/text_encoder.pth',
    'bert_model_path' : '/bert_model.pth',
}

text_model = None
text_tokenizer = None
text_encoder = None
opt = None
image_names = None
reversed_names = None
device = torch.device('cpu')


def home(request):
    '''
    Load all model and features to memory
    '''
    global text_model, text_tokenizer, text_encoder, image_names,reversed_names, opt
    with open(path['option_dict_path'], 'r') as f:
        opt = json.load(f)
    # opt['text_model_pretrained'] = 'bert-base-uncased'
    text_model, text_tokenizer = load_text_model(
        opt['text_model_type'], opt['text_model_pretrained'], opt['output_bert_model'], device, path['bert_model_path'])
    text_encoder = load_transform_model(opt, path['text_encoder_path'], device)
    names_series = []
    reversed_names_series = []
    for feature_file in os.listdir(path['image_feature_folder']):
        name_file = os.path.join(path['image_name_folder'], os.path.splitext(feature_file)[0] + '.csv')
        filenames = pd.Series(pd.read_csv(name_file,header=None, index_col=0).iloc[:,0])
        names_series.append(filenames)
        reversed_names_series.append(pd.Series(filenames.index.values, index=filenames))
    image_names = pd.concat(names_series, ignore_index=True)
    reversed_names = pd.concat(reversed_names)
    return HttpResponse('Setup done!')

def get_images(request, caption, dist_func, k, start_from):
    global text_model, text_tokenizer, text_encoder, image_names, opt
    if dist_func == 'euclide':
        dist_func = euclidean_dist
    else:
        dist_func = cosine_dist
    dists, filenames = get_images_from_caption(caption=caption,
                                              image_features_folder=path['image_feature_folder'],
                                              image_names=image_names,
                                              text_model=text_model,
                                              text_tokenizer=text_tokenizer,
                                              text_encoder=text_encoder,
                                              device=device,
                                              max_seq_len=opt['max_seq_len'],
                                              dist_func=dist_func,
                                              k=k, start_from=start_from)
    response_data = dict()
    response_data['dists'] = dists.tolist()
    response_data['filename'] = filenames
    # print(dists)
    return JsonResponse(response_data)

def query_on_subset(request, caption, dist_func, k, start_from):
    global text_model, text_tokenizer, text_encoder, image_names, reversed_names, opt
    if request.method=="POST":
        if dist_func == 'euclide':
            dist_func = euclidean_dist
        else:
            dist_func = cosine_dist
        subset = request.POST.get('image_list', [])
        dists, filenames = get_images_from_caption_subset(caption=caption,
                                                        subset=subset,
                                                        image_features_folder=path['image_feature_folder'],
                                                        image_names=image_names,
                                                        reversed_names=reversed_names,
                                                        text_model=text_model,
                                                        text_tokenizer=text_tokenizer,
                                                        text_encoder=text_encoder,
                                                        device=device,
                                                        max_seq_len=opt['max_seq_len'],
                                                        dist_func=dist_func,
                                                        k=k, start_from=start_from)
        response_data = dict()
        response_data['dists'] = dists.tolist()
        response_data['filename'] = filenames
        # print(dists)
        return JsonResponse(response_data)
    else:
        return JsonResponse({'dists': [], 'filename': []})