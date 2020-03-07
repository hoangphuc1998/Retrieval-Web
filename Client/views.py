from django.shortcuts import render, HttpResponse
import requests

server_address = 'http://127.0.0.1:8000/server/'

# Create your views here.
def send_image_retrieval_request(caption, dist_func='cosine', k=10):
    url = server_address + caption + '/' + dist_func + '/' + str(k)
    r = requests.get(url=url).json()
    images = r['image']
    dists = r['dists']
    return images, dists

def home(request):
    #images, dists = send_image_retrieval_request('toilet')
    return render(request, 'client_home.html', {'images': range(10)})