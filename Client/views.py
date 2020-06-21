from django.shortcuts import render, HttpResponse
import requests

server_address = 'http://127.0.0.1:8000/server/'

# Create your views here.
def send_image_retrieval_request(caption, dataset, dist_func='cosine', k=10):
    url = server_address + 'query/' + caption + '/' + dist_func + '/' + str(k)+'/0'
    r = requests.get(url=url).json()
    images = r['image']
    dists = r['dists']
    filenames = r['filenames']
    return images, dists, filenames

def home(request):
    return render(request, 'client_home.html', {'caption':''})

def retrieve_images(request):
    print(request.GET.get('dataset'))
    if request.method == 'GET':
        caption = request.GET.get('caption')
        dataset = request.GET.get('dataset')
        images, _, filenames = send_image_retrieval_request(caption,dataset)
        print(filenames)
        return render(request, 'client_query.html', {'images_and_filenames': zip(images, filenames), 'caption': caption})
    else:
        return render(request, 'client_home.html', {'caption':''})