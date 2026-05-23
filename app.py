import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
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
        resume_lower = resume_text.lower()

        text_data = [resume_text, job_desc]

        cv = CountVectorizer()
        matrix = cv.fit_transform(text_data)

        score = cosine_similarity(matrix)[0][1]
        match_percent = round(score * 100, 2)

        # Match Score
        st.subheader("Match Score")
        st.progress(int(match_percent))
        st.success(f"{match_percent}% Match")

        # Graph
        fig, ax = plt.subplots(figsize=(2,2))
        ax.bar(["Match Score"], [match_percent])
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0,101,10))
        ax.tick_params(labelsize=5)
        st.pyplot(fig)

        # ATS Check
        st.subheader("ATS Resume Check")

        ats_keywords = [
            "python",
            "sql",
            "machine learning",
            "data analysis",
            "communication",
            "projects"
        ]

        ats_score = 0

        for word in ats_keywords:
            if word in resume_lower:
                ats_score += 1

        ats_percent = round((ats_score / len(ats_keywords)) * 100, 2)

        st.success(f"ATS Score: {ats_percent}%")

        if ats_percent > 70:
            st.write("ATS Friendly Resume ✅")
        else:
            st.write("Needs ATS Improvement ⚠️")

        # Missing Skills
        skills = [
            "python",
            "machine learning",
            "sql",
            "data analysis",
            "streamlit"
        ]

        missing = []

        for skill in skills:
            if skill not in resume_lower:
                missing.append(skill)

        st.subheader("📌 Missing Skills")

        if missing:
            st.error(missing)
        else:
            st.success("No Missing Skills ✅")

        # Final Feedback
        if match_percent > 70:
            st.balloons()
            st.success("Excellent Match 🚀")
        elif match_percent > 40:
            st.info("Good Match 👍")
        else:
            st.warning("Needs Improvement 📚")

    else:
        st.warning("Upload resume and enter job description")
