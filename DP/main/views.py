from django.shortcuts import render

def index(request):
    context = {
        'title': ['Какво е това?']
    }

    return render(request, 'main/index.html')
