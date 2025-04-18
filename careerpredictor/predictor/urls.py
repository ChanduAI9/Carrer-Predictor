from django.urls import path
from .views import (
    CareerPredictionAPIView,
    home_view,
    upload_resume_view,
    quiz_form_view,
    result_view,
)

urlpatterns = [
    path('predict-career/', CareerPredictionAPIView.as_view(), name='predict-career'),

    path('', home_view, name='home'),
    path('upload/', upload_resume_view, name='upload_resume'),
    path('quiz/', quiz_form_view, name='take_quiz'),
    path('result/', result_view, name='career_result'),
]
