from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from .models import Test
from django.contrib.auth import login, logout, authenticate
from .models import MLMODEL
from django.http import HttpResponse

# Create your views here.


def image_receive(request):
    if request.method == 'POST':
        print('received... something',request.name)
    return HttpResponse('Hello there?')

def index(request):
    # return HttpResponse('Hello, world!')
    return render(
        request=request,
        template_name="myapp/home.html",
        context={'test': Test.objects.all}
    )


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return index(request)
        else:
            for msg in form.error_messages:
                print(form.error_messages[msg])
    else:
        form = UserCreationForm

    return render(request=request,
                  template_name='myapp/register.html',
                  context={'form': form})


def user_login(request):
    return render(request=request,
                  template_name='myapp/user_login.html',
                  context={'test': Test.objects.all})


def ml_model(request):
    return render(request, template_name='myapp/ml_model.html', context={'ml_model':MLMODEL.objects.all})