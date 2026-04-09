from django.urls import path
from .views import *

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', index, name='home'),
    path('old', index_old, name='home_old'),
    path('gpt', index_gpt, name='home_gpt'),
    path('predict/', predict, name='predict'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
