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
image_feature_path = '/home/lehoanganh298/Documents/COCO_val_image_transform.pth'
filename_path = '/home/lehoanganh298/Documents/COCO_filenames.csv'
option_dict_path = '/home/lehoanganh298/Documents/options.pkl'
text_encoder_path = '/home/lehoanganh298/Documents/DualEncoding_Ver2/best/text_encoder.pth'
image_folder = '/home/lehoanganh298/Documents/val2014'

# image_feature_folder = '/media/hoangphuc/Data/LSC_transform'
# filename_folder = '/media/hoangphuc/Data/LSC_filename'
# option_dict_path = '/home/hoangphuc/Documents/options.pkl'
# text_encoder_path = '/home/hoangphuc/Documents/DualEncoding_Ver2/best/text_encoder.pth'
# image_folder = '/media/hoangphuc/Data/LSC2020/lsc2020/Volumes/Samsung_T5/DATASETS/LSC2020'

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
    with open(option_dict_path, 'rb') as f:
        opt = pickle.load(f)
    text_model, text_tokenizer = load_text_model(
        opt['text_model_type'], opt['text_model_pretrained'], device)
    text_encoder = load_transform_model(opt, text_encoder_path, device)
    # image_features, image_names = load_all_feature(
    #     image_feature_path, filename_path, device)
    return HttpResponse('Setup done!')

def get_images(request, caption, dist_func, k, start_from):
    global text_model, text_tokenizer, text_encoder, image_features, image_names
    if dist_func == 'cosine':
        dist_func = cosine_dist
    else:
        dist_func = euclidean_dist
    dists, filenames = get_images_from_caption(caption=caption,
                                              image_features_folder=image_feature_folder,
                                              image_names_folder=filename_folder,
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
        with open(os.path.join(image_folder, filename), 'rb') as f:
            response_data['image'].append(base64.b64encode(f.read()).decode('utf-8'))
    response_data['dists'] = dists.tolist()
    print(dists)
    return JsonResponse(response_data)