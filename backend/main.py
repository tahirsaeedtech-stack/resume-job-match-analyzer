from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from io import BytesIO


# ---------------------------------------------------
# FastAPI app
# ---------------------------------------------------

app = FastAPI(
    title="AI Resume Job Match API",
    version="1.0"
)


# ---------------------------------------------------
# CORS configuration
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Load Sentence Transformer model
# ---------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------
# Skill aliases
# ---------------------------------------------------

skill_aliases = {

    # Programming
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript"],
    "typescript": ["typescript"],
    "c++": ["c++"],
    "c#": ["c#"],

    # Data analysis
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "excel": ["excel", "microsoft excel"],

    # Machine learning
    "machine learning": [
        "machine learning"
    ],

    "scikit-learn": [
        "scikit-learn",
        "sklearn"
    ],

    "tensorflow": [
        "tensorflow"
    ],

    "pytorch": [
        "pytorch"
    ],

    "xgboost": [
        "xgboost"
    ],

    "lightgbm": [
        "lightgbm"
    ],

    "feature engineering": [
        "feature engineering"
    ],

    "model evaluation": [
        "model evaluation",
        "model validation"
    ],

    "cross-validation": [
        "cross-validation",
        "cross validation"
    ],

    "classification": [
        "classification"
    ],

    "regression": [
        "regression"
    ],

    "clustering": [
        "clustering"
    ],

    # AI / NLP
    "artificial intelligence": [
        "artificial intelligence"
    ],

    "nlp": [
        "nlp",
        "natural language processing"
    ],

    "transformers": [
        "transformers",
        "transformer models"
    ],

    "hugging face": [
        "hugging face",
        "huggingface"
    ],

    "sentence transformers": [
        "sentence transformers",
        "sentence-transformers"
    ],

    "embeddings": [
        "embeddings",
        "vector embeddings"
    ],

    "llm": [
        "llm",
        "large language model",
        "large language models"
    ],

    "rag": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation"
    ],

    # Backend / APIs
    "fastapi": [
        "fastapi"
    ],

    "flask": [
        "flask"
    ],

    "django": [
        "django"
    ],

    "rest api": [
        "rest api",
        "restful api",
        "restful services"
    ],

    # Frontend
    "react": [
        "react",
        "react.js",
        "reactjs"
    ],

    "html": [
        "html"
    ],

    "css": [
        "css"
    ],

    # Databases
    "sql": [
        "sql"
    ],

    "postgresql": [
        "postgresql",
        "postgres"
    ],

    "mysql": [
        "mysql"
    ],

    "sqlite": [
        "sqlite"
    ],

    "mongodb": [
        "mongodb",
        "mongo db"
    ],

    # DevOps
    "docker": [
        "docker"
    ],

    "kubernetes": [
        "kubernetes",
        "k8s"
    ],

    "git": [
        "git"
    ],

    "github": [
        "github"
    ],

    "ci/cd": [
        "ci/cd",
        "continuous integration",
        "continuous deployment"
    ],

    # Cloud
    "aws": [
        "aws",
        "amazon web services"
    ],

    "azure": [
        "azure",
        "microsoft azure"
    ],

    "gcp": [
        "gcp",
        "google cloud platform"
    ],

    "cloud deployment": [
        "cloud deployment"
    ],

    # Data engineering
    "spark": [
        "apache spark",
        "spark"
    ],

    "airflow": [
        "apache airflow",
        "airflow"
    ],

    "etl": [
        "etl",
        "extract transform load"
    ],

    # General
    "data analysis": [
        "data analysis",
        "data analytics"
    ],

    "data visualization": [
        "data visualization",
        "data visualisation"
    ],

    "problem solving": [
        "problem solving",
        "problem-solving"
    ]
}

# ---------------------------------------------------
# Skill extraction
# ---------------------------------------------------


def extract_normalized_skills(text):
    text = text.lower()

    found_skills = []

    for canonical_skill, aliases in skill_aliases.items():
        for alias in aliases:
            if alias in text:
                found_skills.append(canonical_skill)
                break

    return sorted(set(found_skills))


# ---------------------------------------------------
# PDF text extraction
# ---------------------------------------------------

def extract_pdf_text(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))

    extracted_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text + "\n"

    return extracted_text


# ---------------------------------------------------
# Resume-job matching logic
# ---------------------------------------------------

def evaluate_match(resume_text, job_description):

    # ---------------------------------------
    # 1. Generate semantic embeddings
    # ---------------------------------------

    resume_embedding = model.encode(
        resume_text,
        convert_to_numpy=True
    )

    job_embedding = model.encode(
        job_description,
        convert_to_numpy=True
    )

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    semantic_score = float(similarity * 100)

    # ---------------------------------------
    # 2. Extract skills
    # ---------------------------------------

    resume_skills = extract_normalized_skills(
        resume_text
    )

    job_skills = extract_normalized_skills(
        job_description
    )

    matched_skills = sorted(
        set(resume_skills) & set(job_skills)
    )

    missing_skills = sorted(
        set(job_skills) - set(resume_skills)
    )

    # ---------------------------------------
    # 3. Calculate skill score
    # ---------------------------------------

    if job_skills:

        skill_match_score = (
            len(matched_skills)
            / len(job_skills)
        ) * 100

        # Hybrid score
        overall_score = (
            semantic_score * 0.60
            + skill_match_score * 0.40
        )

        skill_analysis_available = True

    else:

        # No recognized job skills.
        # Do NOT artificially give the candidate 0%.
        skill_match_score = None

        # Fall back to semantic matching.
        overall_score = semantic_score

        skill_analysis_available = False

    # ---------------------------------------
    # 4. Match level
    # ---------------------------------------

    if overall_score >= 80:

        match_level = "Strong Match"

    elif overall_score >= 65:

        match_level = "Moderate Match"

    else:

        match_level = "Weak Match"

    # ---------------------------------------
    # 5. Recommendation
    # ---------------------------------------

    if not skill_analysis_available:

        recommendation = (
            "The resume shows semantic alignment with this role, "
            "but the current skill taxonomy could not identify enough "
            "explicit technical skills from the job description. "
            "The result is therefore based primarily on semantic similarity."
        )

    elif match_level == "Strong Match":

        recommendation = (
            "Strong alignment with the role. Focus on clearly "
            "demonstrating relevant project impact and measurable results."
        )

    elif match_level == "Moderate Match":

        if missing_skills:

            recommendation = (
                "Good overall fit, but strengthen the resume by "
                "demonstrating experience with: "
                + ", ".join(missing_skills[:4])
                + "."
            )

        else:

            recommendation = (
                "Good overall fit. Strengthen the resume with clearer "
                "evidence of relevant experience and measurable results."
            )

    else:

        if missing_skills:

            recommendation = (
                "The resume has limited alignment with this role. "
                "Key skills to develop or demonstrate include: "
                + ", ".join(missing_skills[:4])
                + "."
            )

        else:

            recommendation = (
                "The resume has limited semantic alignment with this role."
            )

    # ---------------------------------------
    # 6. API response
    # ---------------------------------------

    return {
        "semantic_score": round(
            semantic_score,
            2
        ),

        "skill_match_score": (
            round(float(skill_match_score), 2)
            if skill_match_score is not None
            else None
        ),

        "overall_score": round(
            float(overall_score),
            2
        ),

        "match_level": match_level,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "skill_analysis_available": (
            skill_analysis_available
        ),

        "recommendation": recommendation
    }

    # Create embeddings
    resume_embedding = model.encode(
        resume_text,
        convert_to_numpy=True
    )

    job_embedding = model.encode(
        job_description,
        convert_to_numpy=True
    )

    # Semantic similarity
    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    semantic_score = float(similarity * 100)

    # Extract skills
    resume_skills = extract_normalized_skills(
        resume_text
    )

    job_skills = extract_normalized_skills(
        job_description
    )

    # Matched skills
    matched_skills = sorted(
        set(resume_skills) & set(job_skills)
    )

    # Missing skills
    missing_skills = sorted(
        set(job_skills) - set(resume_skills)
    )

    # Skill match score
    if job_skills:
        skill_match_score = (
            len(matched_skills)
            / len(job_skills)
        ) * 100
    else:
        skill_match_score = 0.0

    # Overall score
    overall_score = (
        semantic_score * 0.6
        + skill_match_score * 0.4
    )

    # Match classification
    if overall_score >= 80:
        match_level = "Strong Match"

    elif overall_score >= 65:
        match_level = "Moderate Match"

    else:
        match_level = "Weak Match"

    # Recommendation
    if match_level == "Strong Match":

        recommendation = (
            "Strong alignment with the role. "
            "Focus on clearly demonstrating relevant project "
            "impact and measurable results."
        )

    elif match_level == "Moderate Match":

        if missing_skills:

            recommendation = (
                "Good overall fit, but strengthen the resume "
                "by demonstrating experience with: "
                + ", ".join(missing_skills[:4])
                + "."
            )

        else:

            recommendation = (
                "Good overall fit. Strengthen the resume with "
                "clearer evidence of relevant experience and "
                "measurable project outcomes."
            )

    else:

        if missing_skills:

            recommendation = (
                "The resume has limited alignment with this role. "
                "Key skills to develop or demonstrate include: "
                + ", ".join(missing_skills[:4])
                + "."
            )

        else:

            recommendation = (
                "The resume has limited semantic alignment "
                "with this role."
            )

    # Final API response
    return {
        "semantic_score": round(
            semantic_score,
            2
        ),
        "skill_match_score": round(
            float(skill_match_score),
            2
        ),
        "overall_score": round(
            float(overall_score),
            2
        ),
        "match_level": match_level,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendation": recommendation
    }


# ---------------------------------------------------
# Home endpoint
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "AI Resume Job Match API is running"
    }


# ---------------------------------------------------
# Health endpoint
# ---------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ---------------------------------------------------
# Analyze resume endpoint
# ---------------------------------------------------

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    # Validate uploaded file
    if resume.content_type != "application/pdf":

        return {
            "error": "Please upload a PDF resume."
        }

    # Read uploaded PDF
    file_bytes = await resume.read()

    # Extract text
    resume_text = extract_pdf_text(
        file_bytes
    )

    # Validate extracted text
    if not resume_text.strip():

        return {
            "error": (
                "Could not extract text from the resume PDF."
            )
        }

    # Validate job description
    if not job_description.strip():

        return {
            "error": (
                "Job description cannot be empty."
            )
        }

    # Analyze match
    result = evaluate_match(
        resume_text,
        job_description
    )

    return result
