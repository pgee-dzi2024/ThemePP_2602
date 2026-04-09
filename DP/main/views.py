from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import Picture
from .ml import predict_image

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB



def index(request):
    pictures = Picture.objects.all()
    context = {
        'title': ['Какво е това?'],
        'pictures': pictures
    }

    return render(request, 'main/index.html', context)


def index_old(request):
    pictures = Picture.objects.all()
    context = {
        'title': ['Какво е това?'],
        'pictures': pictures
    }

    return render(request, 'main/index_old.html', context)


def index_gpt(request):
    pictures = Picture.objects.all()
    context = {
        'title': ['Какво е това?'],
        'pictures': pictures
    }

    return render(request, 'main/index_gpt.html', context)


def _json_error(message, status=400):
    return JsonResponse({
        "success": False,
        "error": message
    }, status=status)


@csrf_exempt
def predict(request):
    if request.method != "POST":
        return _json_error("Само POST заявка е позволена.", status=405)

    uploaded_file = request.FILES.get("image")
    if not uploaded_file:
        return _json_error("Липсва изображение.", status=400)

    if not uploaded_file.content_type or not uploaded_file.content_type.startswith("image/"):
        return _json_error("Невалиден файл. Моля, качете изображение.", status=400)

    if uploaded_file.size > MAX_UPLOAD_SIZE:
        return _json_error("Файлът е твърде голям. Моля, качете изображение до 5 MB.", status=400)

    try:
        result = predict_image(uploaded_file)
    except FileNotFoundError as exc:
        return _json_error(f"Липсва необходим ресурс: {exc}", status=500)
    except ValueError as exc:
        return _json_error(str(exc), status=400)
    except Exception:
        return _json_error("Възникна вътрешна грешка при обработката на изображението.", status=500)

    return JsonResponse({
        "success": True,
        "result": result
    })

