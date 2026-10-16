from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def landing(request):
    return render(request,"land2.html")

def home(request):
    return render(request,'landing.html')