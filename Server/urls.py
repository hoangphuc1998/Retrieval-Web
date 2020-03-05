from django.urls import path
from .views import home, get_images
urlpatterns = [
    path('', home, name='home'),
    path('<str:caption>/<str:dist_func>/<int:k>', get_images, name='get_images')
]
