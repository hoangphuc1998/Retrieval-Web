from django.urls import path
from .views import home, get_images, query_on_subset, query_by_metadata, query_by_metadata_before, query_by_similar_image
# print('-------------------')
# print(home)
# print(path)
urlpatterns = [
    path('', home, name='home'),
    path('query/<str:caption>/<str:dist_func>/<int:k>/<int:start_from>', get_images, name='get_images'),
    path('query_on_subset/<str:caption>/<str:dist_func>/<int:k>/<int:start_from>', query_on_subset, name='query_on_subset'),
    path('query_by_metadata/<str:place>', query_by_metadata, name='query_by_metadata'),
    path('query_by_metadata_before/<str:place>/<int:minute_before>', query_by_metadata_before, name='query_by_metadata_before'),
    path('query_by_similar_image/<int:k>/<int:start_from>', query_by_similar_image, name='query_by_similar_image'),
]
