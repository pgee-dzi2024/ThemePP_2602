from django.shortcuts import render


def index(request):
    context = {
        'title': ['Какво е това?', 'Защо е това?', 'Кога е това?'],
        'button_name': '*'
    }
    return render(request, 'main/index.html', context)
