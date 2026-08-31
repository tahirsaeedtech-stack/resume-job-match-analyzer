# AI Resume–Job Match & Skill Gap Analyzer

An end-to-end AI/NLP application that analyzes the alignment between a candidate's resume and a job description using transformer-based semantic embeddings and explicit skill matching.

The application accepts a **PDF resume** and a **job description**, extracts resume text, calculates semantic similarity, identifies matched and missing skills, and produces an interpretable overall match score with actionable recommendations.

---

## Project Overview

Traditional keyword matching can miss meaningful relationships between a resume and a job description when the wording is different.

This project combines two approaches:

1. **Semantic Matching** — understands contextual similarity using transformer embeddings.
2. **Skill Gap Analysis** — identifies explicit technical skills shared between the resume and job description.

The result is a hybrid resume–job alignment system that provides more information than a simple keyword matcher.

---

## Application Preview

![AI Resume Job Match Analyzer](docs/images/resume-match-result.png)

---

## Key Features

- PDF resume upload
- Automatic PDF text extraction
- Transformer-based sentence embeddings
- Semantic similarity scoring
- Technical skill extraction and normalization
- Matched skill identification
- Missing skill / skill-gap analysis
- Hybrid overall match score
- Strong / Moderate / Weak match classification
- Actionable resume recommendations
- FastAPI REST backend
- React + Vite frontend
- Responsive user interface
- Privacy-conscious design with no resume persistence

---

## AI / NLP Approach

### 1. Resume Text Extraction

The uploaded PDF resume is processed using `pypdf`.

The extracted text becomes the input to the NLP matching pipeline.

```text
PDF Resume
    ↓
Text Extraction
    ↓
Resume Text
```

---

### 2. Transformer Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

from the Sentence Transformers ecosystem.

The resume and job description are transformed into dense numerical vector representations.

```text
Resume Text ───────→ Sentence Transformer ───────→ Resume Embedding

Job Description ──→ Sentence Transformer ───────→ Job Embedding
```

These embeddings capture semantic information beyond exact keyword overlap.

---

### 3. Semantic Similarity

Cosine similarity is calculated between the resume embedding and job-description embedding.

Conceptually:

```text
Resume Embedding
       ↓
Cosine Similarity
       ↑
Job Embedding
```

A higher similarity indicates stronger semantic alignment between the supplied texts.

The semantic score is an **alignment score**, not a probability that a candidate will be hired.

---

### 4. Skill Extraction

The application also maintains a normalized technical skill taxonomy covering areas such as:

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- PyTorch
- Machine Learning
- NLP
- Transformers
- Hugging Face
- Embeddings
- LLMs
- RAG
- FastAPI
- Flask
- Django
- React
- SQL
- PostgreSQL
- MongoDB
- Docker
- Kubernetes
- Git
- GitHub
- AWS
- Azure
- GCP
- Spark
- Airflow
- ETL
- Data Analysis
- Data Visualization

Aliases are normalized into canonical skill names before comparison.

The system then determines:

```text
Matched Skills = Resume Skills ∩ Job Skills

Missing Skills = Job Skills - Resume Skills
```

---

## Hybrid Match Score

When explicit job skills are successfully recognized, the overall score combines:

```text
60% Semantic Similarity
+
40% Skill Coverage
```

Conceptually:

```text
Overall Match =
(0.60 × Semantic Score)
+
(0.40 × Skill Match Score)
```

For the current MVP, match levels are interpreted as:

| Overall Score | Match Level |
|---|---|
| 80% and above | Strong Match |
| 65% – 79.99% | Moderate Match |
| Below 65% | Weak Match |

These weights and thresholds are **MVP heuristics** rather than scientifically calibrated hiring thresholds.

### Skill-analysis fallback

If the current skill taxonomy cannot identify sufficient explicit skills from the job description, the application does **not** artificially assign a 0% skill score.

Instead:

- Skill Match is shown as `N/A`
- The overall result falls back to semantic similarity
- The UI explains that explicit skill analysis was unavailable

This prevents missing taxonomy coverage from unfairly lowering the displayed alignment score.

---

## Recommendation Engine

The application generates a concise recommendation based on:

- overall match level
- semantic alignment
- detected missing skills
- availability of skill-gap analysis

For example, a moderate match may recommend strengthening evidence for specific missing technical skills.

The recommendation layer is deterministic and is kept separate from the transformer-based semantic scoring.

---

## System Architecture

```text
                    ┌───────────────────────┐
                    │     React Frontend    │
                    │                       │
                    │  Resume PDF Upload    │
                    │  Job Description      │
                    └───────────┬───────────┘
                                │
                                │ HTTP / FormData
                                ▼
                    ┌───────────────────────┐
                    │    FastAPI Backend    │
                    └───────────┬───────────┘
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
    ┌──────────────────┐                  ┌──────────────────┐
    │ PDF Text         │                  │ Job Description  │
    │ Extraction       │                  │ Processing       │
    └────────┬─────────┘                  └────────┬─────────┘
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
                 ┌─────────────────────────────┐
                 │ Sentence Transformer Model  │
                 │ all-MiniLM-L6-v2            │
                 └──────────────┬──────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Semantic Similarity   │
                    └───────────┬───────────┘
                                │
             ┌──────────────────┴──────────────────┐
             │                                     │
             ▼                                     ▼
    ┌──────────────────┐                  ┌──────────────────┐
    │ Resume Skill     │                  │ Job Skill        │
    │ Extraction       │                  │ Extraction       │
    └────────┬─────────┘                  └────────┬─────────┘
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Skill Gap Analysis    │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Hybrid Scoring        │
                    │ + Recommendation      │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ React Results UI      │
                    └───────────────────────┘
```

---

## Technology Stack

### AI / NLP

- Sentence Transformers
- Hugging Face model ecosystem
- Scikit-learn
- NumPy
- Cosine similarity

### Backend

- Python
- FastAPI
- Uvicorn
- pypdf
- python-multipart

### Frontend

- React
- Vite
- JavaScript
- CSS

---

## Project Structure

```text
resume-job-match-analyzer/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── index.css
│   ├── package.json
│   └── ...
│
├── notebooks/
│   └── 01_semantic_matching.ipynb
│
├── data/
│   └── samples/
│
├── docs/
│   └── images/
│       └── resume-match-result.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Analyze Resume

```http
POST /analyze
```

Inputs:

- `resume` — PDF resume
- `job_description` — job-description text

Example response:

```json
{
  "semantic_score": 76.42,
  "skill_match_score": 71.43,
  "overall_score": 74.42,
  "match_level": "Moderate Match",
  "matched_skills": [
    "python",
    "machine learning",
    "scikit-learn",
    "fastapi"
  ],
  "missing_skills": [
    "docker",
    "aws"
  ],
  "skill_analysis_available": true,
  "recommendation": "Good overall fit, but strengthen the resume by demonstrating experience with: docker, aws."
}
```

---

## Local Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd resume-job-match-analyzer
```

### 2. Create a Python virtual environment

Windows:

```powershell
python -m venv .venv
```

### 3. Install backend dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Start the FastAPI backend

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Install frontend dependencies

Open another terminal:

```powershell
cd frontend
npm install
```

### 6. Start the React frontend

```powershell
npm run dev
```

The frontend will typically run at:

```text
http://localhost:5173
```

---

## Validation

The matching pipeline was manually validated using deliberately different resume/job pairs representing:

- strong alignment
- moderate alignment
- weak/unrelated alignment

The baseline tests showed the expected ordering, with closely aligned ML profiles receiving substantially higher scores than unrelated profiles.

This validation demonstrates basic behavioral consistency but should **not** be interpreted as a formal benchmark of hiring accuracy.

---

## Important Limitations

This project is an AI/NLP portfolio application and has several important limitations.

### Skill taxonomy

Skill extraction currently uses a curated taxonomy and aliases. Skills outside this taxonomy may not be recognized.

### Long documents

`all-MiniLM-L6-v2` has a limited input length. Very long resumes or job descriptions may require section-based or chunk-based embedding strategies for more complete semantic representation.

### Heuristic scoring

The 60/40 weighting and Strong/Moderate/Weak thresholds are application-level heuristics and have not been calibrated against real hiring outcomes.

### PDF extraction

The current implementation is designed primarily for text-based PDF resumes. Image-only or scanned resumes may require OCR.

### Semantic similarity is not hiring probability

A high semantic score means the supplied resume text is semantically similar to the supplied job description.

It does **not** mean that the candidate has the same probability of being hired.

---

## Responsible AI & Fairness

This application is designed as a **resume–job alignment and skill-gap analysis tool**, not an automated hiring or rejection system.

It does not intentionally use or score protected personal characteristics.

The system should not be used as a substitute for human judgment in employment decisions.

Results can also reflect:

- limitations in the skill taxonomy
- wording differences
- missing resume information
- biases or omissions present in job descriptions
- limitations of the underlying embedding model

The application therefore presents its outputs as alignment indicators rather than employment decisions.

---

## Privacy

Resumes may contain sensitive personal information.

The application is designed to process uploaded resume content for analysis without intentionally persisting the uploaded PDF.

Real candidate resumes should never be committed to the public repository.

The `.gitignore` configuration excludes PDF files and local upload directories to reduce the risk of accidentally publishing resume data.

---

## Future Improvements

Potential extensions include:

- section-aware resume parsing
- chunked semantic embeddings for long documents
- larger and externally maintained skill taxonomies
- experience-level matching
- education and certification extraction
- weighted required vs preferred skills
- batch resume ranking
- CrossEncoder reranking
- configurable job-specific skill weights
- calibrated evaluation using a larger labeled dataset
- cloud deployment

---

## What This Project Demonstrates

This project demonstrates practical experience with:

- Natural Language Processing
- Transformer-based embeddings
- Semantic similarity
- Feature/scoring design
- Skill extraction and normalization
- Responsible AI considerations
- FastAPI REST API development
- React frontend development
- PDF processing
- Full-stack AI application integration

---

## Disclaimer

This project is intended for educational, portfolio, and candidate-facing resume analysis purposes.

The generated scores represent textual and skill alignment between supplied documents. They should not be interpreted as predictions of hiring success or as automated employment decisions.