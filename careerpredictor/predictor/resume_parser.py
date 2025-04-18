import fitz  # PyMuPDF

def extract_resume_text(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_features_from_text(text):
    features = {
        "coding_skills_rating": 8 if "Python" in text else 5,
        "communication_skills": 7 if "team" in text.lower() else 5,
        "public_speaking_score": 6 if "presentation" in text.lower() else 4,
        "logical_quotient_rating": 9 if "algorithm" in text.lower() else 6,
        "teamwork_rating": 8 if "collaborate" in text.lower() else 5,
        "interested_area": "Technical" if "backend" in text.lower() or "api" in text.lower() else "Management",
        "career_goal": "MNC Job",
        "preferred_work_environment": "Team",
        "internships_completed": text.lower().count("intern"),
        "mini_projects_done": text.lower().count("project"),
        "coding_contests_participated": 1,
        "age": 22,
        "gender": "Male",
        "graduation_year": 2025,
        "interested_type_of_books": "Tech",
        "management_or_technical": "Technical",
        "mbti_type": "INTJ",
        "preferred_time": "Morning",
        "personality_trait": "Introvert",
        "operating_system_score": 85,
        "mathematics_score": 90,
        "dbms_score": 88,
    }
    return features
