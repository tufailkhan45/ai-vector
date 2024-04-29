from django.urls import path
from . import chatbot

urlpatterns = [
    path('api/getMessage', chatbot.getMessage, name='getMessage'),
]