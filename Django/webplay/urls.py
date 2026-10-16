from django.urls import path
from . import views

#URLconfig
urlpatterns=[
    path('', views.landing),
    path('home', views.home, name='home')
]
