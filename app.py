import streamlit as st
import pandas as pd
from io import BytesIO
import docx
import PyPDF2
from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_similarity_model()

DEFAULT_SKILLS = [
    "Basic Accounting and Bookkeeping", "Data Entry and Management", "Customer Service Skills",
    "Communication Skills", "Time Management", "Sales and Marketing Basics", "Problem-Solving Skills",
    "Financial Literacy", "Computer Literacy", "Organizational Skills", "Business Ethics", "Interpersonal Skills",
    "Project Management Basics", "Administrative Skills", "Professionalism", "Analytical Skills", "Adaptability",
    "Inventory Management", "Conflict Resolution", "Financial Software Familiarity"
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

def extract_responsibilities(text):
    """Extract responsibilities from text by finding relevant sections."""
    lines = text.split("\n")
    responsibilities = []
    capture = False
    for line in lines:
        if "responsibilities" in line.lower() or "duties" in line.lower():
            capture = True
        elif capture and line.strip() == "":
            break  # Stop capturing at the next blank line
        elif capture:
            responsibilities.append(line.strip())
    return responsibilities

def match_skills(responsibilities, skills):
    """Match responsibilities with skills using semantic similarity."""
    responsibilities_str = " ".join(responsibilities)
    skills_embeddings = model.encode(skills, convert_to_tensor=True)
    responsibilities_embedding = model.encode(responsibilities_str, convert_to_tensor=True)
    similarities = util.cos_sim(responsibilities_embedding, skills_embeddings).flatten()
    matched_skills = [skills[i] for i in range(len(skills)) if similarities[i] > 0.5]  # Threshold for similarity
    return matched_skills

st.title("Job Description Skills Matcher")

uploaded_file = st.file_uploader("Choose a file (.pdf, .docx, or .txt)", type=["pdf", "docx", "txt"])
custom_skills_input = st.text_area("Enter custom skills (comma-separated) or leave blank for default:", "")
skills_list = [s.strip() for s in custom_skills_input.split(",")] if custom_skills_input else DEFAULT_SKILLS

if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    responsibilities = extract_responsibilities(text)
    matched_skills = match_skills(responsibilities, skills_list)
    
    st.subheader("Concise Job Responsibilities:")
    st.write(responsibilities if responsibilities else "No responsibilities found.")
    
    st.subheader("Matched Skills:")
    st.write(", ".join(matched_skills) if matched_skills else "No matched skills.")
    
    df = pd.DataFrame({
        "Responsibilities": ["; ".join(responsibilities)],
        "Matched Skills": [", ".join(matched_skills)]
    })
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.download_button("Download Results as Excel", output.getvalue(), "results.xlsx")
