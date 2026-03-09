from django.shortcuts import render
from .models import Picture


def index(request):
    pictures = Picture.objects.all()
    context = {
        'title': ['Какво е това?'],
        'pictures': pictures
    }

    return render(request, 'main/index.html', context)
