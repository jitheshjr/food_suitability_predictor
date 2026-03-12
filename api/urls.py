from django.urls import path
from .views import test_api, predict

urlpatterns = [
    path('test/', test_api),
    path('predict/', predict),
]