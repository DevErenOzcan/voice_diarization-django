from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('save_audio/', views.save_audio, name='save_audio'),
    path('transcribe_audio/', views.transcribe_audio, name='transcribe_audio'),
]
