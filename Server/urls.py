from django.urls import path
from .views import *
urlpatterns = [
    path('', home, name='home'),
    path('query/<str:caption>/<str:dist_func>/<int:k>/<int:start_from>', get_images, name='get_images'),
    path('query_on_subset/<str:caption>/<str:dist_func>/<int:k>/<int:start_from>', query_on_subset, name='query_on_subset'),
]
