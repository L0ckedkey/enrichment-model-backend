from django.urls import path
from . import views

urlpatterns = [
    path('hello', views.Hello),
    path('dimension', views.Dimension),
    path('subdimension', views.Subdimension)
]