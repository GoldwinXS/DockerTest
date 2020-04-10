from django.shortcuts import render
from django.http import HttpResponse
from .models import Test


# Create your views here.

def index(request):
    return HttpResponse('Hello, world!')
    # return render(
    #     request=request,
    #     template_name="index.html",
    #     context={'test': Test}
    # )
