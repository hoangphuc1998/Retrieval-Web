from django.urls import path
# from .views import home, get_images, query_on_subset, query_by_metadata, query_by_metadata_before, query_by_similar_image
from .views import query_by_caption, \
                    query_by_metadata, \
                    query_by_caption_on_subset,\
                    query_by_metadata_on_subset,\
                    query_by_time_range_on_subset,\
                    query_images_before,\
                    query_similar_images,\
                    query_adjacent_images\

urlpatterns = [
    # path('', home, name='home'),
    path('query_by_caption/<str:caption>/<str:dist_func>/<int:num_images>', query_by_caption, name='get_images'),
    path('query_by_caption_on_subset', query_by_caption_on_subset, name='query_on_subset'),
    
    path('query_by_metadata/<str:places>', query_by_metadata, name='query_by_metadata'),
    path('query_by_metadata_on_subset', query_by_metadata_on_subset, name='query_by_metadata_on_subset'),
    
    path('query_by_time_range_on_subset',query_by_time_range_on_subset, name='query_by_time_range_on_subset'),
    path('query_images_before', query_images_before, name='query_images_before'),

    path('query_similar_images/<str:image>/<int:num_images>', query_similar_images, name='query_by_similar_image'),
    path('query_adjacent_images/<str:image>/<int:num_images',query_adjacent_images, name='query_adjacent_images')
]

# urlpatterns = [
#     path('', home, name='home'),
#     path('query/<str:caption>/<str:dist_func>/<int:k>/<int:start_from>', get_images, name='get_images'),
#     path('query_on_subset/<str:caption>/<str:dist_func>/<int:k>/<int:start_from>', query_on_subset, name='query_on_subset'),
#     path('query_by_metadata/<str:place>', query_by_metadata, name='query_by_metadata'),
#     path('query_by_metadata_before/<str:place>/<int:minute_before>', query_by_metadata_before, name='query_by_metadata_before'),
#     path('query_by_similar_image/<int:k>/<int:start_from>', query_by_similar_image, name='query_by_similar_image'),
# ]
