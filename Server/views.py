from django.shortcuts import render
import torch
from .utils import *
import pickle
import json
import os
import pandas as pd
from django.http import JsonResponse, HttpResponse
import base64
from django.views.decorators.csrf import csrf_exempt
import glob

# Create your views here.

# Global variables
path = {
    'image_feature_folder' : '/home/lehoanganh298/Projects/models/LSC_Attention_Ver9/feature',
    'filename_folder' : '/home/lehoanganh298/Projects/models/LSC_Attention_Ver9/filename',
    'option_dict_path' : '/home/lehoanganh298/Projects/models/LSC_Attention_Ver9/options.json',
    'text_encoder_path' :  '/home/lehoanganh298/Projects/models/LSC_Attention_Ver9/text_encoder.pth',
    'bert_model_path' : '/home/lehoanganh298/Projects/models/LSC_Attention_Ver9/bert_model.pth',
    'metadata_path': '/home/lehoanganh298/Projects/datasets/LSC2020/lsc2020-metadata.csv',
    'concepts_path': '/home/lehoanganh298/Projects/datasets/LSC2020/concepts.csv',
    'similar_feature_folder': '/dataset/LSC/similar_feature/',
    'similar_filename_folder': '/dataset/LSC/similar_filename/',
}

text_model = None
text_tokenizer = None
text_encoder = None
opt = None
image_names = None
reversed_names = None
metadata = None
concepts = None
device = torch.device('cpu')


def query_by_caption(request, caption, dist_func, num_images):
    filename_df = pd.read_csv(path['filename_folder']+'/2015-02-23.csv')
    return JsonResponse({'filenames': list(filename_df.iloc[:,1])[:num_images]})

def query_by_caption_on_subset(request):
    """
    data: {
        subset: subset,
        caption: caption,
        numImages: numImages
      }
    """
    if request.method=="POST":
        data = json.loads(request.body.decode('utf-8'))
        return JsonResponse({'filenames': data['subset'][0::2][:data['numImages']]})

def query_by_metadata(request, places):
    filename_df = pd.read_csv(path['filename_folder']+'/2015-02-24.csv')
    return JsonResponse({'filenames': list(filename_df.iloc[:,1][:200])})

def query_by_metadata_on_subset(request):
    """
    data: {
        subset: subset,
        locations: seperated by '&'
      }
    """
    if request.method=="POST":
        data = json.loads(request.body.decode('utf-8'))
        return JsonResponse({'filenames': data['subset'][1::3]})
        
def query_similar_images(request, image, num_images):
    ''' image: <folder_name>&<file_name>'''
    folder_name,image_name = image.split('&')
    filename_df = pd.read_csv(path['filename_folder']+'/'+folder_name+'.csv')
    return JsonResponse({'filenames': list(filename_df.iloc[:,1])[1000:1000+num_images]})

# def query_by_metadata_before(request, place, minute_before):

# def home(request):
#     '''
#     Load all model and features to memory
#     '''
#     global text_model, text_tokenizer, text_encoder, image_names,reversed_names, opt, metadata, concepts
#     with open(path['option_dict_path'], 'r') as f:
#         opt = json.load(f)
#     # opt['text_model_pretrained'] = 'bert-base-uncased'

#     text_model, text_tokenizer = load_text_model(
#         opt['text_model_type'], opt['text_model_pretrained'], opt['output_bert_model'], device, path['bert_model_path'])
#     text_encoder = load_transform_model(opt, path['text_encoder_path'], device)
#     names_series = []
#     reversed_names_series = []
#     for feature_file in os.listdir(path['image_feature_folder']):
#         name_file = os.path.join(path['filename_folder'], os.path.splitext(feature_file)[0] + '.csv')
#         filenames = pd.Series(pd.read_csv(name_file,header=None, index_col=0).iloc[:,0])
#         names_series.append(filenames)
#         reversed_names_series.append(pd.Series(filenames.index.values, index=filenames))
#     image_names = pd.concat(names_series, ignore_index=True)
#     reversed_names_series = pd.concat(reversed_names_series)
#     # Query with metadata setup
#     metadata = pd.read_csv(path['metadata_path'])
#     metadata = metadata[['minute_id', 'semantic_name']]
#     metadata = metadata.dropna()
#     concepts = pd.read_csv(path['concepts_path'])
#     concepts = concepts[['minute_id', 'image_path']]
#     concepts['image_path'] = concepts['image_path'].str.slice(17)
#     metadata = metadata.merge(concepts)
#     metadata['date'] = metadata['minute_id'].str.slice(0,8)
#     metadata['hour'] = metadata['minute_id'].str.slice(9,11)
#     metadata['minute'] = metadata['minute_id'].str.slice(11)

#     return HttpResponse('Setup done!')

# def get_images(request, caption, dist_func, k, start_from):
#     global text_model, text_tokenizer, text_encoder, image_names, opt
#     if dist_func == 'euclide':
#         dist_func = euclidean_dist
#     else:
#         dist_func = cosine_dist
#     dists, filenames = get_images_from_caption(caption=caption,
#                                               image_features_folder=path['image_feature_folder'],
#                                               image_names=image_names,
#                                               text_model=text_model,
#                                               text_tokenizer=text_tokenizer,
#                                               text_encoder=text_encoder,
#                                               device=device,
#                                               max_seq_len=opt['max_seq_len'],
#                                               dist_func=dist_func,
#                                               k=k, start_from=start_from)
#     response_data = dict()
#     response_data['filenames'] = filenames
#     #TODO: Add different image dataset
#     # for filename in filenames:
#     #     with open(os.path.join(path[dataset]['image_folder'], filename), 'rb') as f:
#     #         response_data['image'].append(base64.b64encode(f.read()).decode('utf-8'))
#     response_data['dists'] = dists.tolist()
#     return JsonResponse(response_data)

# def query_on_subset(request, caption, dist_func, k, start_from):
#     global text_model, text_tokenizer, text_encoder, image_names, reversed_names, opt
#     if request.method=="POST":
#         if dist_func == 'euclide':
#             dist_func = euclidean_dist
#         else:
#             dist_func = cosine_dist
#         subset = request.POST.get('image_list', [])
#         dists, filenames = get_images_from_caption_subset(caption=caption,
#                                                         subset=subset,
#                                                         image_features_folder=path['image_feature_folder'],
#                                                         image_names=image_names,
#                                                         reversed_names=reversed_names,
#                                                         text_model=text_model,
#                                                         text_tokenizer=text_tokenizer,
#                                                         text_encoder=text_encoder,
#                                                         device=device,
#                                                         max_seq_len=opt['max_seq_len'],
#                                                         dist_func=dist_func,
#                                                         k=k, start_from=start_from)
#         response_data = dict()
#         response_data['dists'] = dists.tolist()
#         response_data['filename'] = filenames
#         # print(dists)
#         return JsonResponse(response_data)
#     else:
#         return JsonResponse({'dists': [], 'filename': []})

# def query_by_metadata(request, place):
#     global metadata
#     res = metadata.loc[metadata['semantic_name'].str.lower().str.contains(place)]['image_path']
#     response_data = dict()
#     response_data['filename'] = res.tolist()
#     return JsonResponse(response_data)

# def query_by_metadata_before(request, place, minute_before):
#     global metadata, concepts
#     res = metadata.loc[metadata['semantic_name'].str.lower().str.contains(place)]
#     image_set = set()
#     if len(res)>0:
#         begin = res.iloc[0]
#         minute_id = begin['minute_id']
#         begin_time = datetime.datetime(int(minute_id[:4]), int(minute_id[4:6]), int(minute_id[6:8]), int(minute_id[9:11]), int(minute_id[11:]))
        
#         image_set = image_set.union(get_image_set_before_time(concepts, minute_id, minute_before))
#         for _, row in res.iterrows():
#             minute_id = row['minute_id']
#             time = datetime.datetime(int(minute_id[:4]), int(minute_id[4:6]), int(minute_id[6:8]), int(minute_id[9:11]), int(minute_id[11:]))
#             if time <= begin_time + datetime.timedelta(minutes=1):
#                 begin_time = time
#                 continue
#             else:
#                 begin_time = time
#                 image_set = image_set.union(get_image_set_before_time(concepts, minute_id, minute_before))
#     response_data = dict()
#     response_data['filename'] = list(image_set)
#     return JsonResponse(response_data)

# def query_by_similar_image(request, k, start_from):
#     global path, device
#     if request.method=="POST":
#         image_path = request.POST.get('filename', '')
#         if len(image_path) == 0:
#             return JsonResponse({'dists': [], 'filename': []})
#         dists, filenames = get_similar_images(image_path=image_path,
#                                                 similar_feature_folder=path['similar_feature_folder'],
#                                                 similar_filename_folder=path['similar_filename_folder'],
#                                                 device=device, k=k, start_from=start_from)
#         response_data = dict()
#         response_data['dists'] = dists.tolist()
#         response_data['filename'] = filenames
#         return JsonResponse(response_data)
#     else:
#         return JsonResponse({'dists': [], 'filename': []})
