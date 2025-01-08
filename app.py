import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO
import docx
import PyPDF2

# Load summarization model
summarizer = pipeline("summarization")

# Define the top 20 skills or load a custom list
DEFAULT_SKILLS = [
    "Basic Accounting and Bookkeeping", "Data Entry and Management", "Customer Service Skills", 
    "Communication Skills", "Time Management", "Sales and Marketing Basics", 
    "Problem-Solving Skills", "Financial Literacy", "Computer Literacy", "Organizational Skills", 
    "Business Ethics", "Interpersonal Skills", "Project Management Basics", "Administrative Skills", 
    "Professionalism", "Analytical Skills", "Adaptability", "Inventory Management", 
    "Conflict Resolution", "Financial Software Familiarity"
]

def extract_text_from_file(uploaded_file):
    """Extract text from DOCX, PDF, or TXT files."""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8")

def summarize_text(text):
    """Summarize the extracted text."""
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

def match_skills(text, skills):
    """Match job description text with skills."""
    matched = [skill for skill in skills if skill.lower() in text.lower()]
    return matched

# Streamlit App Layout
st.title("Job Description Skills Matcher")
st.write("Upload a job description file to extract and match skills.")

# File uploader
uploaded_file = st.file_uploader("Choose a file (.pdf, .docx, or .txt)", type=["pdf", "docx", "txt"])

# Custom skills list
custom_skills_input = st.text_area("Enter custom skills (comma-separated) or leave blank for default:", "")

# Load custom or default skills
skills_list = [s.strip() for s in custom_skills_input.split(",")] if custom_skills_input else DEFAULT_SKILLS

if uploaded_file:
    # Extract and summarize text
    text = extract_text_from_file(uploaded_file)
    summary = summarize_text(text)
    
    # Match skills
    matched_skills = match_skills(text, skills_list)
    
    # Display results
    st.subheader("Summary of Job Description:")
    st.write(summary)
    
    st.subheader("Matched Skills:")
    st.write(", ".join(matched_skills))
    
    # Option to download results
    df = pd.DataFrame({
        "Job Responsibilities": [summary],
        "Matched Skills": [", ".join(matched_skills)]
    })
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.download_button("Download Matched Skills as Excel", output.getvalue(), "matched_skills.xlsx")
