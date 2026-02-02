import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- UI Settings -----------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>AI Resume Analyzer & Job Matcher</h1>
    <p style='text-align: center;'>Upload your resume and a job description to see match score, missing skills, and AI suggestions.</p>
    """,
    unsafe_allow_html=True
)

# ----------------- Upload Resume & JD -----------------
with st.sidebar:
    st.header("Upload Inputs")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste Job Description here")

# ----------------- Skills list -----------------
skills_list = [
    "python", "machine learning", "deep learning", "sql", "excel",
    "data analysis", "nlp", "streamlit", "git", "docker", "aws"
]

# ----------------- Functions -----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def calculate_match(resume_text, jd_text):
    documents = [resume_text, jd_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(documents)
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def detect_missing_skills(resume_text, jd_text, skills):
    resume_lower = resume_text.lower()
    missing = []
    for skill in skills:
        if skill not in resume_lower and skill in jd_text.lower():
            missing.append(skill)
    return missing

def generate_ai_suggestions(missing_skills, match_percentage):
    suggestions = []
    if match_percentage < 70:
        suggestions.append("Consider tailoring your resume to match the job description more closely.")
    if missing_skills:
        suggestions.append("Focus on learning or highlighting missing skills: " + ", ".join(missing_skills))
    suggestions.append("Make sure your resume highlights measurable achievements and projects.")
    return suggestions

# ----------------- Main Logic -----------------
if resume_file and job_description:
    resume_text = extract_text_from_pdf(resume_file)

    match_percentage = calculate_match(resume_text, job_description)
    missing_skills = detect_missing_skills(resume_text, job_description, skills_list)
    ai_suggestions = generate_ai_suggestions(missing_skills, match_percentage)

    # ----------------- Display Results -----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Match Score")
        st.progress(match_percentage / 100)
        st.write(f"**Match Percentage:** {match_percentage}%")

    with col2:
        if missing_skills:
            st.subheader("⚠️ Missing Skills")
            st.write(", ".join(missing_skills))
        else:
            st.subheader("✅ All key skills found!")

    st.subheader("💡 AI Suggestions")
    for suggestion in ai_suggestions:
        st.write(f"- {suggestion}")

    st.subheader("📄 Resume Text Preview (first 800 chars)")
    st.text(resume_text[:800])
