from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import Picture
from .ml import predict_image


def index(request):
    pictures = Picture.objects.all()
    context = {
        'title': ['Какво е това?'],
        'pictures': pictures
    }

    return render(request, 'main/index.html', context)


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Само POST заявка е позволена."}, status=405)

    uploaded_file = request.FILES.get("image")
    if not uploaded_file:
        return JsonResponse({"error": "Липсва изображение."}, status=400)

    result = predict_image(uploaded_file)

    return JsonResponse({
        "success": True,
        "result": result
    })

