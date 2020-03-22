from django.shortcuts import render, HttpResponse
import requests

server_address = 'http://127.0.0.1:8000/server/'

# Create your views here.
def send_image_retrieval_request(caption, dist_func='cosine', k=10):
    url = server_address + caption + '/' + dist_func + '/' + str(k)+'/0'
    r = requests.get(url=url).json()
    images = r['image']
    dists = r['dists']
    return images, dists

def home(request):
    return render(request, 'client_home.html', {'caption':''})

def retrieve_images(request):
    if request.method == 'GET':
        caption = request.GET.get('caption')
        images, _ = send_image_retrieval_request(caption)
        return render(request, 'client_query.html', {'images': images, 'caption': caption})
    else:
        return render(request, 'client_home.html', {'caption':''})