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

# Default list of skills (can be customized by the user)
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
    """
    Extracts the responsibilities section from the job description.
    It looks for keywords like "responsibilities" and stops when it finds "qualifications" or "how to apply".
    """
    lower_text = text.lower()
    start_index = lower_text.find("responsibilities")
    if start_index == -1:
        return ["No clear 'Responsibilities' section found."]
    
    # Try to find an end marker (like the next section header)
    end_index = lower_text.find("qualifications", start_index)
    if end_index == -1:
        end_index = lower_text.find("how to apply", start_index)
    if end_index == -1:
        end_index = len(text)
    
    # Extract the block of text
    block = text[start_index:end_index]
    # Remove any colon or header text and split by newlines
    lines = block.split("\n")
    responsibilities = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Remove the header line if it contains 'responsibilities'
        if "responsibilities" in stripped.lower():
            parts = stripped.split(":", 1)
            if len(parts) > 1 and parts[1].strip():
                responsibilities.append(parts[1].strip())
        else:
            responsibilities.append(stripped)
    return responsibilities

def match_skills(responsibilities, skills):
    """
    Matches the responsibilities text against the provided skills list using semantic similarity.
    A lower threshold is used to capture more loosely related matches.
    """
    responsibilities_str = " ".join(responsibilities)
    skills_embeddings = model.encode(skills, convert_to_tensor=True)
    responsibilities_embedding = model.encode(responsibilities_str, convert_to_tensor=True)
    similarities = util.cos_sim(responsibilities_embedding, skills_embeddings).flatten()
    
    # Set a lower threshold to capture more matches
    threshold = 0.3
    matched_skills = [skills[i] for i in range(len(skills)) if similarities[i] > threshold]
    return matched_skills

# Streamlit App Layout
st.title("Job Description Skills Matcher")
st.write("Upload a job description file to extract and match skills.")

# File uploader for PDFs, DOCX, or TXT files
uploaded_file = st.file_uploader("Choose a file (.pdf, .docx, or .txt)", type=["pdf", "docx", "txt"])

# Allow user to provide a custom skills list
custom_skills_input = st.text_area("Enter custom skills (comma-separated) or leave blank for default:", "")
skills_list = [s.strip() for s in custom_skills_input.split(",")] if custom_skills_input else DEFAULT_SKILLS

if uploaded_file:
    # Extract text from the uploaded file
    text = extract_text_from_file(uploaded_file)
    # Extract the responsibilities using our updated function
    responsibilities = extract_responsibilities(text)
    # Match skills based on the responsibilities
    matched_skills = match_skills(responsibilities, skills_list)
    
    st.subheader("Concise Job Responsibilities:")
    st.write(responsibilities if responsibilities else "No responsibilities found.")
    
    st.subheader("Matched Skills:")
    st.write(", ".join(matched_skills) if matched_skills else "No matched skills.")
    
    # Prepare output Excel file
    df = pd.DataFrame({
        "Responsibilities": ["; ".join(responsibilities)],
        "Matched Skills": [", ".join(matched_skills)]
    })
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.download_button("Download Results as Excel", output.getvalue(), "results.xlsx")
