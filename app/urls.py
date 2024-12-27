from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('transcribe_audio/', views.transcribe_audio, name='transcribe_audio'),
    path('person_labeling/', views.person_labeling, name='person_labeling'),
]
