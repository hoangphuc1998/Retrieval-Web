from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import *
from .apps import ServerConfig
import json
import numpy as np
# Create your views here.

def home(request):
    '''
    Load all model and features to memory
    '''
    return HttpResponse("Server is running")

def query_by_caption(request, caption, dist_func, num_images):
    if dist_func == 'euclide':
        dist_func = euclidean_dist
    else:
        dist_func = cosine_dist
    dists, filenames = get_images_from_caption(caption=caption,
                                              image_features_folder=ServerConfig.path['sajem_feature_folder'],
                                              image_names=ServerConfig.image_names,
                                              text_model=ServerConfig.text_model,
                                              text_tokenizer=ServerConfig.text_tokenizer,
                                              text_encoder=ServerConfig.text_encoder,
                                              device=ServerConfig.device,
                                              max_seq_len=ServerConfig.opt['max_seq_len'],
                                              dist_func=dist_func,
                                              k=num_images,
                                              start_from=0)
    response_data = dict()
    response_data['filenames'] = filenames
    response_data['dists'] = dists.tolist()
    return JsonResponse(response_data)

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
        dists, filenames = get_images_from_caption_subset(caption=data['caption'],
                                                        subset=data['subset'],
                                                        image_features_folder=ServerConfig.path['sajem_feature_folder'],
                                                        image_names=ServerConfig.image_names,
                                                        reversed_names=ServerConfig.reversed_names_series,
                                                        text_model=ServerConfig.text_model,
                                                        text_tokenizer=ServerConfig.text_tokenizer,
                                                        text_encoder=ServerConfig.text_encoder,
                                                        device=ServerConfig.device,
                                                        max_seq_len=ServerConfig.opt['max_seq_len'],
                                                        dist_func=cosine_dist,
                                                        k=int(data['numImages']),
                                                        start_from=0)
        response_data = dict()
        response_data['filenames'] = filenames
        response_data['dists'] = dists.tolist()
        return JsonResponse(response_data)
    else:
        return JsonResponse({'dists': [], 'filename': []})

def query_by_metadata_on_subset(request):
    """
    data: {
        subset: subset,
        locations: seperated by '|'
      }
    """
    if request.method=="POST":
        data = json.loads(request.body.decode('utf-8'))
        metadata = ServerConfig.metadata
        places = data['locations'].split('|')
        res = metadata.loc[(metadata['image_path'].isin(data['subset'])) & (metadata['semantic_name'].isin(places))]
        sorterIndex = dict(zip(data['subset'],range(len(data['subset']))))
        res['rank'] = res['image_path'].map(sorterIndex)
        res = res.sort_values('rank', ascending=True)
        response_data = dict()
        response_data['filenames'] = res['image_path'].tolist()
        return JsonResponse(response_data)
    else:
        return JsonResponse({'filenames': []})

def query_by_time_range_on_subset(request):
    """
    data: {
        subset: subset,
        timeBegin:"07:30",
        timeEnd:"08:30"
      }
    """
    if request.method=="POST":
        data = json.loads(request.body.decode('utf-8'))
        concepts = ServerConfig.concepts
        time_begin_str = ''.join(data['timeBegin'].split(':'))
        time_end_str = ''.join(data['timeEnd'].split(':'))
        res = concepts.loc[(concepts['minute_id'].str.slice(9,13).astype(str)>=time_begin_str) 
                            & (concepts['minute_id'].str.slice(9,13).astype(str)<=time_end_str)
                            & (concepts['image_path'].isin(data['subset']))]
        sorterIndex = dict(zip(data['subset'],range(len(data['subset']))))
        res['rank'] = res['image_path'].map(sorterIndex)
        res = res.sort_values('rank', ascending=True)
        response_data = dict()
        response_data['filenames'] = res['image_path'].tolist()
        return JsonResponse(response_data)
    else:
        return JsonResponse({'filenames': []})

# def query_by_time_range(request, begin_time, end_time):
#     concepts = ServerConfig.concepts
#     time_begin_str = ''.join(begin_time.split(':'))
#     time_end_str = ''.join(end_time.split(':'))
#     res = concepts.loc[(concepts['minute_id'].str.slice(9,13).astype(str)>=time_begin_str) 
#                         & (concepts['minute_id'].str.slice(9,13).astype(str)<=time_end_str)]
#     response_data = dict()
#     response_data['filenames'] = res['image_path'].tolist()
#     return JsonResponse(response_data)



def query_by_time(request):
    '''
    timeBegin, timeEnd: hour and minute (hh:MM)
    dowBegin, dowEnd: day of week (0-6s, Sunday will be 0)
    dayBegin, dayEnd: day of month (1-31)
    monthBegin, monthEnd: month (1-12)
    yearBegin, yearEnd: year (yyyy)
    subset: list of previous step's images
    All: -1 for default
    
    '''
    if request.method == "POST":
        data = json.loads(request.body.decode('utf-8'))
        concepts = ServerConfig.concepts
        query_cons = []
        if data['timeBegin']!=-1:
            time_begin_str = ''.join(data['timeBegin'].split(':'))
            time_end_str = ''.join(data['timeEnd'].split(':'))
            query_cons.append(f'(minute_id.str.slice(9,13).astype(\'str\') >= \"{time_begin_str}\")')
            query_cons.append(f'(minute_id.str.slice(9,13).astype(\'str\') <= \"{time_end_str}\")')
        # Day of week
        dowBegin = int(data['dowBegin'])
        dowEnd = int(data['dowEnd'])
        if dowBegin >=0 and dowBegin <=6:
            dowBegin = (dowBegin - 1) % 7
            dowEnd = (dowEnd - 1) % 7
            query_cons.append(f'(dow >= {dowBegin})')
            query_cons.append(f'(dow <= {dowEnd})')
        # Day of month
        dayBegin = int(data['dayBegin'])
        dayEnd = int(data['dayEnd'])
        if dayBegin >= 1:
            query_cons.append(f'(day >= {dayBegin})')
            query_cons.append(f'(day <= {dayEnd})')
        # Month
        monthBegin = int(data['monthBegin'])
        monthEnd = int(data['monthEnd'])
        if monthBegin >= 1:
            query_cons.append(f'(month >= {monthBegin})')
            query_cons.append(f'(month <= {monthEnd})')
        # Year
        yearBegin = int(data['yearBegin'])
        yearEnd = int(data['yearEnd'])
        if yearBegin >= 1:
            query_cons.append(f'(year >= {yearBegin})')
            query_cons.append(f'(year <= {yearEnd})')
        # Query
        image_list = concepts.query(' & '.join(query_cons))['image_path'].tolist()
        res = image_list
        if len(data['subset']) > 0:
            #res = [x for x in data['subset'] if x in image_list]
            subset = np.array(data['subset'])
            res = subset[np.isin(subset, np.array(image_list))].tolist()
        return JsonResponse({'filenames': res})
    else:
        return JsonResponse({'filenames': []})

def query_images_before(request):
    """
    data: {
        subset: subset,
        minutes:30,
      }
    """
    if request.method=="POST":
        data = json.loads(request.body.decode('utf-8'))
        metadata = ServerConfig.metadata
        concepts = ServerConfig.concepts
        minute_before = int(data['minutes'])
        res = metadata.loc[metadata['image_path'].isin(data['subset'])]
        image_set = set()
        if len(res)>0:
            begin = res.iloc[0]
            minute_id = begin['minute_id']
            begin_time = datetime.datetime(int(minute_id[:4]), int(minute_id[4:6]), int(minute_id[6:8]), int(minute_id[9:11]), int(minute_id[11:]))
            
            image_set = image_set.union(get_image_set_before_time(concepts, minute_id, minute_before))
            for _, row in res.iterrows():
                minute_id = row['minute_id']
                time = datetime.datetime(int(minute_id[:4]), int(minute_id[4:6]), int(minute_id[6:8]), int(minute_id[9:11]), int(minute_id[11:]))
                if time <= begin_time + datetime.timedelta(minutes=1):
                    begin_time = time
                    continue
                else:
                    begin_time = time
                    image_set = image_set.union(get_image_set_before_time(concepts, minute_id, minute_before))
        response_data = dict()
        response_data['filenames'] = list(image_set)
        return JsonResponse(response_data)
    else:
        return JsonResponse({'filenames': []})


def query_similar_images(request, image, num_images):
    ''' image: <folder_name>&<file_name>'''
    image_path = image.replace('&', '/')
    if len(image_path) == 0:
        return JsonResponse({'dists': [], 'filename': []})
    dists, filenames = get_similar_images(image_path=image_path,
                                            similar_feature_folder=ServerConfig.path['resnet_feature_folder'],
                                            image_names=ServerConfig.image_names,
                                            reversed_names=ServerConfig.reversed_names_series,
                                            device=ServerConfig.device, k=num_images, start_from=0)
    response_data = dict()
    response_data['dists'] = dists.tolist()
    response_data['filenames'] = filenames
    return JsonResponse(response_data)

def query_by_metadata(request, places):
    metadata = ServerConfig.metadata
    places = places.split('|')
    res = metadata.loc[metadata['semantic_name'].isin(places)]['image_path']
    response_data = dict()
    response_data['filenames'] = res.tolist()
    return JsonResponse(response_data)