from django.urls import path
from .views import home, retrieve_images
urlpatterns = [
    path('', home, name='home'),
    path('query/', retrieve_images, name='query')
]
