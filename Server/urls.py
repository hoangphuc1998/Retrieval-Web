from django.urls import path
# from .views import home, get_images, query_on_subset, query_by_metadata, query_by_metadata_before, query_by_similar_image
from .views import *

urlpatterns = [
    path('query_by_caption/<str:caption>/<str:dist_func>/<int:num_images>', query_by_caption, name='get_images'),
    path('query_by_caption_on_subset', query_by_caption_on_subset, name='query_on_subset'),
    
    path('query_by_metadata/<str:places>', query_by_metadata, name='query_by_metadata'),
    path('query_by_metadata_on_subset', query_by_metadata_on_subset, name='query_by_metadata_on_subset'),

    path('query_by_time_range/<str:begin_time>/<str:end_time>', query_by_time_range, name='query_by_time_range'),
    path('query_by_time_range_on_subset',query_by_time_range_on_subset, name='query_by_time_range_on_subset'),
    path('query_images_before', query_images_before, name='query_images_before'),

    path('query_similar_images/<str:image>/<int:num_images>', query_similar_images, name='query_by_similar_image'),
    # path('query_adjacent_images/<str:image>/<int:num_images',query_adjacent_images, name='query_adjacent_images'),
]
