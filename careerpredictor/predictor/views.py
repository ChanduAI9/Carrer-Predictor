from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ResumeUploadSerializer
from .resume_parser import extract_resume_text, extract_features_from_text
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os, json
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

# Load model and utilities
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = load_model(os.path.join(BASE_DIR, 'models', 'career_model.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))
career_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'career_data.csv'))
X_ref = pd.get_dummies(career_data.drop('career_path', axis=1))
ref_columns = X_ref.columns

def generate_reasoning(predicted_label, resume_text=""):
    lines = []
    if predicted_label == "Software Tester":
        if "automation" in resume_text.lower():
            lines.append("You have experience with automation processes, essential for software testing.")
        if "ci/cd" in resume_text.lower() or "jenkins" in resume_text.lower():
            lines.append("You mentioned CI/CD tools like Jenkins, often used in testing pipelines.")
        if "bug" in resume_text.lower() or "testing" in resume_text.lower():
            lines.append("You referenced testing or debugging experience relevant to QA roles.")
        lines.append("Your academic work demonstrates detail orientation and logical reasoning.")
        lines.append("Your resume reflects traits valued in software quality analysis roles.")
    elif predicted_label == "Data Scientist":
        lines.append("You’ve worked on data projects and shown analytical thinking.")
        lines.append("Experience with Python and SQL aligns with data science workflows.")
        lines.append("Your resume highlights ML and NLP, key for data science roles.")
        lines.append("You understand data pipelines and predictive modeling.")
        lines.append("Your academic and project work show suitability for data-focused roles.")
    return lines[:5]

class CareerPredictionAPIView(APIView):
    def post(self, request):
        serializer = ResumeUploadSerializer(data=request.data)
        if serializer.is_valid():
            resume_file = request.FILES.get("resume", None)
            quiz_answers = serializer.validated_data.get("quiz_answers")

            if resume_file:
                text = extract_resume_text(resume_file)
                features = extract_features_from_text(text)
            elif quiz_answers:
                features = quiz_answers
                text = ""
            else:
                return Response({"error": "Either resume or quiz_answers must be provided."}, status=status.HTTP_400_BAD_REQUEST)

            input_df = pd.DataFrame([features])
            input_encoded = pd.get_dummies(input_df).reindex(columns=ref_columns, fill_value=0)
            input_scaled = scaler.transform(input_encoded)
            pred_probs = model.predict(input_scaled)
            pred_class = np.argmax(pred_probs)
            pred_label = label_encoder.inverse_transform([pred_class])[0]
            reason = generate_reasoning(pred_label, text)

            return Response({
                "predicted_career_path": pred_label,
                "confidence": f"{np.max(pred_probs) * 100:.2f}%",
                "reason": reason
            })

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    


def home_view(request):
    return render(request, 'predictor/home.html')

def upload_resume_view(request):
    if request.method == 'POST' and request.FILES.get('resume'):
        from .resume_parser import extract_resume_text, extract_features_from_text
        text = extract_resume_text(request.FILES['resume'])
        features = extract_features_from_text(text)

        input_df = pd.DataFrame([features])
        input_encoded = pd.get_dummies(input_df).reindex(columns=ref_columns, fill_value=0)
        input_scaled = scaler.transform(input_encoded)
        pred_probs = model.predict(input_scaled)
        pred_class = np.argmax(pred_probs)
        pred_label = label_encoder.inverse_transform([pred_class])[0]
        reason = generate_reasoning(pred_label, text)

        return render(request, 'predictor/result.html', {
            'predicted_career_path': pred_label,
            'confidence': f"{np.max(pred_probs) * 100:.2f}%",
            'reason': reason
        })

    return render(request, 'predictor/upload_resume.html')

def quiz_form_view(request):
    return render(request, 'predictor/quiz_form.html')

def result_view(request):
    return render(request, 'predictor/result.html')
