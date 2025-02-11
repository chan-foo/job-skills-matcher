import streamlit as st
import pandas as pd
from io import BytesIO
import docx
import PyPDF2
import re
from sentence_transformers import SentenceTransformer, util

# Load the similarity model and cache it to speed up processing
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_similarity_model()

# Define skills lists for different two-year diploma programs
program_options = {
    "Two-Year Business Diploma": [
        "Basic Accounting and Bookkeeping", "Data Entry and Management", "Customer Service Skills", 
        "Communication Skills", "Time Management", "Sales and Marketing Basics", "Problem-Solving Skills", 
        "Financial Literacy", "Computer Literacy", "Organizational Skills", "Business Ethics", 
        "Interpersonal Skills", "Project Management Basics", "Administrative Skills", "Professionalism", 
        "Analytical Skills", "Adaptability", "Inventory Management", "Conflict Resolution", 
        "Financial Software Familiarity"
    ],
    "Two-Year Administrative Diploma": [
        "Office Administration", "Customer Service", "Communication Skills", "Data Entry", 
        "Records Management", "Time Management", "Microsoft Office Proficiency", "Team Coordination", 
        "Organizational Skills", "Problem Solving", "Professionalism"
    ]
    # You can add more programs as needed.
}

# Let the user select the diploma program (or use a custom skills list)
program_choice = st.selectbox("Select your diploma program", list(program_options.keys()))
skills_list = program_options[program_choice]

# Option to override with a custom skills list if desired
custom_skills_input = st.text_area("Or enter a custom skills list (comma-separated) instead:", "")
if custom_skills_input.strip():
    skills_list = [s.strip() for s in custom_skills_input.split(",") if s.strip()]

# Function to extract text from various file types
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8")

# Improved extraction of responsibilities using a regex-based approach
def extract_responsibilities(text):
    # This regex looks for text following "Role Responsibilities" until another likely header appears.
    pattern = re.compile(
        r"(?:Role Responsibilities(?:\s*[:\-])?)(.*?)(?=QUALIFICATIONS|Requirements|How To Apply|$)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        block = match.group(1).strip()
        # Split the block into separate lines or bullet points
        responsibilities = [line.strip() for line in re.split(r"[\n\.]+", block) if line.strip()]
        return responsibilities
    else:
        return []

# Semantic skill matching function
def match_skills(responsibilities, skills, threshold=0.2):
    responsibilities_str = " ".join(responsibilities)
    skills_embeddings = model.encode(skills, convert_to_tensor=True)
    responsibilities_embedding = model.encode(responsibilities_str, convert_to_tensor=True)
    similarities = util.cos_sim(responsibilities_embedding, skills_embeddings).flatten()
    # For debugging: display similarity scores for each skill
    st.write("Similarity scores:", {skill: float(similarities[i]) for i, skill in enumerate(skills)})
    matched_skills = [skills[i] for i in range(len(skills)) if similarities[i] > threshold]
    return matched_skills

# Streamlit app layout
st.title("Job Description Skills Matcher")
st.write("Upload a job description file (.pdf, .docx, or .txt) to see which diploma-acquired skills match.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    responsibilities = extract_responsibilities(text)
    matched_skills = match_skills(responsibilities, skills_list, threshold=0.3)
    
    st.subheader("Concise Job Responsibilities:")
    if responsibilities:
        st.write(responsibilities)
    else:
        st.write("No clear responsibilities section found.")
    
    st.subheader("Matched Skills:")
    if matched_skills:
        st.write(", ".join(matched_skills))
    else:
        st.write("No matched skills based on the selected diploma program's skill set.")
    
    # Optional: Generate an Excel file for download with the results
    df = pd.DataFrame({
        "Responsibilities": ["; ".join(responsibilities)],
        "Matched Skills": [", ".join(matched_skills)]
    })
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.download_button("Download Results as Excel", output.getvalue(), "results.xlsx")
