import streamlit as st
import pandas as pd
from io import BytesIO
import docx
import PyPDF2
import re
from sentence_transformers import SentenceTransformer, util

# Load the semantic similarity model and cache it to speed up processing
@st.cache_resource
def load_similarity_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_similarity_model()

# Define skills lists for different two-year diploma programs
program_options = {
    "Two-Year Business Diploma": [
        "Financial accounting", "Managerial accounting", "Financial analysis", 
        "Cost control and budgeting", "Taxation principles", "Corporate finance", 
        "Strategic management", "Business law and risk management", "Contract and employment law", 
        "Marketing strategy", "Human resource management", "Operations and supply chain management", 
        "Project management", "Statistical analysis", "Business data visualization", 
        "Economic forecasting", "Business decision-making", "Enterprise resource planning (ERP)", 
        "Customer relationship management (CRM)", "Supply chain management (SCM)", 
        "Digital marketing", "Business ethics and corporate governance", 
        "Intrapreneurship and innovation", "Organizational behavior", "Leadership development", 
        "Strategic planning", "Performance evaluation", "Business simulations", 
        "Critical thinking", "Problem-solving", "Effective communication", 
        "Professional writing", "Leadership and team collaboration", 
        "Decision-making under pressure", "Adaptability and innovation", 
        "Ethical reasoning", "Time management", "Organizational skills", 
        "Negotiation and conflict resolution", "Customer relationship management", 
        "Networking and stakeholder engagement", "Cultural competency", 
        "Presentation and public speaking", "Resilience and stress management", 
        "Emotional intelligence", "Analytical reasoning", "Proactive decision-making"
    ],
    "Two-Year Administrative Diploma": [
        "Office Administration", "Customer Service", "Communication Skills", "Data Entry",
        "Records Management", "Time Management", "Microsoft Office Proficiency", 
        "Team Coordination", "Organizational Skills", "Problem Solving", "Professionalism"
    ]
}

# Let the user select the diploma program
program_choice = st.selectbox("Select your diploma program", list(program_options.keys()))
skills_list = program_options[program_choice]

# Option to override with a custom skills list
custom_skills_input = st.text_area("Or enter a custom skills list (comma-separated) instead:", "")
if custom_skills_input.strip():
    skills_list = [s.strip() for s in custom_skills_input.split(",") if s.strip()]

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8")

# Function to extract responsibilities from the job description text
def extract_responsibilities(text):
    # This regex captures text following either "Role Responsibilities" or "Responsibilities"
    pattern = re.compile(
        r"(?:Role\s+Responsibilities|Responsibilities)(?:\s*[:\-])?(.*?)(?=Qualifications|Requirements|How To Apply|$)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if match:
        block = match.group(1).strip()
        # Split the block by newlines or periods to create a list of bullet points
        responsibilities = [line.strip() for line in re.split(r"[\n\.]+", block) if line.strip()]
        return responsibilities
    else:
        return []

# Function to perform semantic matching between responsibilities and skills
def match_skills(responsibilities, skills, threshold=0.15):
    responsibilities_str = " ".join(responsibilities)
    skills_embeddings = model.encode(skills, convert_to_tensor=True)
    responsibilities_embedding = model.encode(responsibilities_str, convert_to_tensor=True)
    similarities = util.cos_sim(responsibilities_embedding, skills_embeddings).flatten()
    
    # Debug output: display similarity scores for each skill
    st.write("Similarity scores:", {skills[i]: float(similarities[i]) for i in range(len(skills))})
    
    matched_skills = [skills[i] for i in range(len(skills)) if similarities[i] > threshold]
    return matched_skills

# Streamlit App Layout
st.title("Job Description Skills Matcher")
st.write("Upload a job description file (.pdf, .docx, or .txt) to see which diploma-acquired skills match.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    responsibilities = extract_responsibilities(text)
    
    st.subheader("Extracted Responsibilities")
    if responsibilities:
        st.write(" ".join(responsibilities))
    else:
        st.write("No responsibilities section found.")
    
    matched_skills = match_skills(responsibilities, skills_list, threshold=0.15)
    
    st.subheader("Matched Skills:")
    if matched_skills:
        st.write(", ".join(matched_skills))
    else:
        st.write("No matched skills based on the selected diploma program's skill set.")
    
    # Generate an Excel file with two columns:
    # Column 1: Full diploma skills (from the selected program)
    # Column 2: "Yes" if the skill was matched, blank otherwise
    job_required = ["Yes" if skill in matched_skills else "" for skill in skills_list]
    
    df_excel = pd.DataFrame({
        "Diploma Skills": skills_list,
        "Job Required Skills": job_required
    })
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_excel.to_excel(writer, index=False)
    st.download_button("Download Comparison Results as Excel", output.getvalue(), "results_comparison.xlsx")
