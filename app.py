import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Job Matcher", layout="wide")

st.title("🚀 Smart Job Resume Matcher")
st.write("Check how well your resume matches the job role")

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

resume = st.file_uploader("📄 Upload Resume PDF", type="pdf")
job_desc = st.text_area("💼 Paste Job Description")

if st.button("🔍 Match Resume"):

    if resume and job_desc:

        resume_text = read_pdf(resume)

        text_data = [resume_text, job_desc]

        cv = CountVectorizer()
        matrix = cv.fit_transform(text_data)

        score = cosine_similarity(matrix)[0][1]
        match_percent = round(score * 100, 2)

        st.subheader("Match Score")
        st.progress(int(match_percent))
        st.success(f"{match_percent}% Match")

        skills = ["python", "machine learning", "sql", "data analysis", "streamlit"]

        resume_lower = resume_text.lower()

        missing = []

        for skill in skills:
            if skill not in resume_lower:
                missing.append(skill)

        st.subheader("📌 Missing Skills")

        if missing:
            st.error(missing)
        else:
            st.success("No Missing Skills ✅")

        if match_percent > 70:
            st.balloons()
            st.success("Excellent Match 🚀")
        elif match_percent > 40:
            st.info("Good Match 👍")
        else:
            st.warning("Needs Improvement 📚")

    else:
        st.warning("Upload resume and enter job description")