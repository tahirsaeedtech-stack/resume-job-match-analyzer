import logging
import re
from io import BytesIO
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

API_VERSION = "2.3.0"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_FILE_SIZE = 5 * 1024 * 1024

CHUNK_SIZE_WORDS = 70
CHUNK_OVERLAP_WORDS = 15

TOP_EVIDENCE_RESULTS = 2


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="AI Resume Job Match API",
    description=(
        "Transformer-powered Resume–Job Matching API with semantic "
        "similarity, requirement-aware skill analysis, resume section "
        "understanding, semantic evidence retrieval, and explainable "
        "candidate-job matching."
    ),
    version=API_VERSION,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Requirement weights
# ============================================================

REQUIREMENT_WEIGHTS = {
    "required": 1.0,
    "preferred": 0.65,
    "optional": 0.35,
}


# ============================================================
# Load Sentence Transformer
# ============================================================

logger.info("Loading Sentence Transformer model...")

model = SentenceTransformer(MODEL_NAME)

logger.info("Sentence Transformer model loaded.")


# ============================================================
# Skill aliases
# ============================================================

SKILL_ALIASES = {

    # --------------------------------------------------------
    # Programming
    # --------------------------------------------------------

    "python": [
        "python",
    ],

    "java": [
        "java",
    ],

    "javascript": [
        "javascript",
        "js",
    ],

    "typescript": [
        "typescript",
    ],

    "c++": [
        "c++",
    ],

    "c#": [
        "c#",
    ],

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    "pandas": [
        "pandas",
    ],

    "numpy": [
        "numpy",
    ],

    "matplotlib": [
        "matplotlib",
    ],

    "seaborn": [
        "seaborn",
    ],

    "excel": [
        "excel",
        "microsoft excel",
    ],

    "sql": [
        "sql",
    ],

    "postgresql": [
        "postgresql",
        "postgres",
    ],

    "mysql": [
        "mysql",
    ],

    "sqlite": [
        "sqlite",
    ],

    "mongodb": [
        "mongodb",
        "mongo db",
    ],

    # --------------------------------------------------------
    # Machine Learning
    # --------------------------------------------------------

    "machine learning": [
        "machine learning",
        "ml",
    ],

    "scikit-learn": [
        "scikit-learn",
        "scikit learn",
        "sklearn",
    ],

    "tensorflow": [
        "tensorflow",
    ],

    "pytorch": [
        "pytorch",
    ],

    "xgboost": [
        "xgboost",
    ],

    "lightgbm": [
        "lightgbm",
    ],

    "catboost": [
        "catboost",
    ],

    "feature engineering": [
        "feature engineering",
        "feature creation",
        "feature extraction",
    ],

    "model evaluation": [
        "model evaluation",
        "model validation",
        "evaluate models",
        "evaluating models",
    ],

    "cross-validation": [
        "cross-validation",
        "cross validation",
        "k-fold",
        "k fold",
    ],

    "classification": [
        "classification",
        "classifier",
        "classifiers",
    ],

    "regression": [
        "regression",
        "regressor",
        "regressors",
    ],

    "clustering": [
        "clustering",
    ],

    "predictive modeling": [
        "predictive modeling",
        "predictive modelling",
    ],

    "statistics": [
        "statistics",
        "statistical analysis",
    ],

    # --------------------------------------------------------
    # AI / NLP
    # --------------------------------------------------------

    "artificial intelligence": [
        "artificial intelligence",
        "ai",
    ],

    "nlp": [
        "nlp",
        "natural language processing",
    ],

    "transformers": [
        "transformers",
        "transformer model",
        "transformer models",
        "transformer-based",
    ],

    "hugging face": [
        "hugging face",
        "huggingface",
    ],

    "sentence transformers": [
        "sentence transformers",
        "sentence-transformers",
        "sentence transformer",
    ],

    "embeddings": [
        "embedding",
        "embeddings",
        "sentence embedding",
        "sentence embeddings",
        "vector embedding",
        "vector embeddings",
    ],

    "semantic search": [
        "semantic search",
        "semantic similarity",
        "semantic retrieval",
    ],

    "llm": [
        "llm",
        "llms",
        "large language model",
        "large language models",
    ],

    "rag": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
    ],

    "prompt engineering": [
        "prompt engineering",
    ],

    "computer vision": [
        "computer vision",
    ],

    # --------------------------------------------------------
    # Backend
    # --------------------------------------------------------

    "fastapi": [
        "fastapi",
    ],

    "flask": [
        "flask",
    ],

    "django": [
        "django",
    ],

    "rest api": [
        "rest api",
        "rest APIs",
        "restful api",
        "restful APIs",
        "restful services",
    ],

    # --------------------------------------------------------
    # Frontend
    # --------------------------------------------------------

    "react": [
        "react",
        "react.js",
        "reactjs",
    ],

    "html": [
        "html",
        "html5",
    ],

    "css": [
        "css",
        "css3",
    ],

    # --------------------------------------------------------
    # DevOps / MLOps
    # --------------------------------------------------------

    "docker": [
        "docker",
        "dockerfile",
    ],

    "kubernetes": [
        "kubernetes",
        "k8s",
    ],

    "git": [
        "git",
    ],

    "github": [
        "github",
    ],

    "ci/cd": [
        "ci/cd",
        "continuous integration",
        "continuous deployment",
        "github actions",
    ],

    "mlflow": [
        "mlflow",
    ],

    "mlops": [
        "mlops",
        "machine learning operations",
    ],

    # --------------------------------------------------------
    # Cloud
    # --------------------------------------------------------

    "aws": [
        "aws",
        "amazon web services",
    ],

    "azure": [
        "azure",
        "microsoft azure",
    ],

    "gcp": [
        "gcp",
        "google cloud",
        "google cloud platform",
        "cloud run",
    ],

    "cloud deployment": [
        "cloud deployment",
        "cloud computing",
        "deployed to cloud",
        "cloud deployed",
    ],

    # --------------------------------------------------------
    # Vector search
    # --------------------------------------------------------

    "pgvector": [
        "pgvector",
    ],

    "faiss": [
        "faiss",
    ],

    "pinecone": [
        "pinecone",
    ],

    # --------------------------------------------------------
    # Data engineering
    # --------------------------------------------------------

    "spark": [
        "spark",
        "apache spark",
    ],

    "airflow": [
        "airflow",
        "apache airflow",
    ],

    "etl": [
        "etl",
        "extract transform load",
    ],

    # --------------------------------------------------------
    # General technical abilities
    # --------------------------------------------------------

    "data analysis": [
        "data analysis",
        "data analytics",
    ],

    "data visualization": [
        "data visualization",
        "data visualisation",
    ],

    "problem solving": [
        "problem solving",
        "problem-solving",
    ],
}


# ============================================================
# Resume section aliases
# ============================================================

SECTION_ALIASES = {

    "summary": [
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career summary",
        "objective",
        "career objective",
        "about me",
    ],

    "skills": [
        "skills",
        "technical skills",
        "core skills",
        "core competencies",
        "technical expertise",
        "technologies",
        "technical stack",
        "tech stack",
        "tools and technologies",
    ],

    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "career history",
    ],

    "projects": [
        "projects",
        "project experience",
        "personal projects",
        "academic projects",
        "professional projects",
        "key projects",
        "selected projects",
        "portfolio projects",
        "machine learning projects",
        "ai projects",
    ],

    "education": [
        "education",
        "academic background",
        "academic qualifications",
        "educational background",
        "qualifications",
        "education and training",
        "education & training",
        "education and technical training",
        "education & technical training",
    ],

    "certifications": [
        "certifications",
        "certification",
        "certificates",
        "professional certifications",
        "technical certifications",
        "courses",
        "training",
        "professional training",
    ],
}


# ============================================================
# Requirement indicators
# ============================================================

REQUIRED_INDICATORS = [
    "required",
    "must",
    "must have",
    "should have",
    "need to have",
    "minimum requirement",
    "essential",
    "mandatory",
    "expected to",
]

PREFERRED_INDICATORS = [
    "preferred",
    "preferably",
    "nice to have",
    "strong plus",
    "advantage",
    "desirable",
]

OPTIONAL_INDICATORS = [
    "beneficial",
    "bonus",
    "helpful",
    "optional",
    "would be useful",
    "is a plus",
]


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text: str) -> str:

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# Heading normalization
# ============================================================

def normalize_heading(text: str) -> str:

    text = text.lower()

    text = text.replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"[^a-z0-9+# ]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Safe alias matching
# ============================================================

def contains_alias(
    text: str,
    alias: str,
) -> bool:

    pattern = (
        rf"(?<!\w)"
        rf"{re.escape(alias.lower())}"
        rf"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            text.lower(),
        )
    )


# ============================================================
# Skill extraction
# ============================================================

def extract_normalized_skills(
    text: str
) -> List[str]:

    found_skills = set()

    normalized_text = text.lower()

    for canonical_skill, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            if contains_alias(
                normalized_text,
                alias,
            ):

                found_skills.add(
                    canonical_skill
                )

                break

    return sorted(
        found_skills
    )


# ============================================================
# PDF extraction
# ============================================================

def extract_pdf_text(
    file_bytes: bytes
) -> str:

    try:

        reader = PdfReader(
            BytesIO(file_bytes)
        )

        pages = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                pages.append(
                    page_text
                )

        return normalize_text(
            "\n".join(pages)
        )

    except Exception as exc:

        logger.exception(
            "PDF extraction failed."
        )

        raise ValueError(
            "Unable to read the uploaded PDF."
        ) from exc


# ============================================================
# Determine whether line looks like section heading
# ============================================================

def looks_like_heading(
    line: str
) -> bool:

    stripped = line.strip()

    if not stripped:
        return False

    words = stripped.split()

    # Resume headings are normally short.
    if len(words) > 8:
        return False

    if len(stripped) > 80:
        return False

    letters = [
        char
        for char in stripped
        if char.isalpha()
    ]

    if not letters:
        return False

    uppercase_ratio = (
        sum(
            char.isupper()
            for char in letters
        )
        / len(letters)
    )

    if uppercase_ratio >= 0.65:
        return True

    normalized = normalize_heading(
        stripped
    )

    for aliases in SECTION_ALIASES.values():

        for alias in aliases:

            normalized_alias = normalize_heading(
                alias
            )

            if normalized == normalized_alias:
                return True

    return False


# ============================================================
# Detect section from heading
# ============================================================

def identify_section_heading(
    line: str
) -> Optional[str]:

    if not looks_like_heading(
        line
    ):
        return None

    normalized_line = normalize_heading(
        line
    )

    # --------------------------------------------------------
    # Exact alias matching first
    # --------------------------------------------------------

    for section_name, aliases in SECTION_ALIASES.items():

        for alias in aliases:

            if (
                normalized_line
                == normalize_heading(alias)
            ):
                return section_name

    # --------------------------------------------------------
    # Flexible heading recognition
    # --------------------------------------------------------

    tokens = set(
        normalized_line.split()
    )

    # Projects should have priority where explicitly present.
    if "project" in tokens or "projects" in tokens:
        return "projects"

    if (
        "education" in tokens
        or "academic" in tokens
        or "qualification" in tokens
        or "qualifications" in tokens
    ):
        return "education"

    if (
        "certification" in tokens
        or "certifications" in tokens
        or "certificate" in tokens
        or "certificates" in tokens
    ):
        return "certifications"

    if (
        "experience" in tokens
        or "employment" in tokens
    ):
        return "experience"

    if (
        "skill" in tokens
        or "skills" in tokens
        or "competencies" in tokens
    ):
        return "skills"

    if (
        "summary" in tokens
        or "profile" in tokens
        or "objective" in tokens
    ):
        return "summary"

    # Special case:
    # EDUCATION & TECHNICAL TRAINING
    if (
        "education" in normalized_line
        and "training" in normalized_line
    ):
        return "education"

    return None


# ============================================================
# Resume section detection
# ============================================================

def detect_resume_sections(
    resume_text: str
) -> Dict[str, str]:

    lines = [
        line.strip()
        for line in resume_text.splitlines()
        if line.strip()
    ]

    sections = {
        "general": []
    }

    current_section = "general"

    for line in lines:

        detected_section = (
            identify_section_heading(
                line
            )
        )

        if detected_section:

            current_section = (
                detected_section
            )

            sections.setdefault(
                current_section,
                [],
            )

            continue

        sections.setdefault(
            current_section,
            [],
        ).append(
            line
        )

    final_sections = {}

    for section_name, content in sections.items():

        joined = " ".join(
            content
        ).strip()

        if joined:

            final_sections[
                section_name
            ] = joined

    return final_sections


# ============================================================
# Batch text embedding
# ============================================================

def encode_texts(
    texts: List[str]
):

    if not texts:
        return None

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )


# ============================================================
# Semantic similarity
# ============================================================

def calculate_semantic_similarity(
    text_a: str,
    text_b: str,
) -> float:

    embeddings = encode_texts(
        [
            text_a,
            text_b,
        ]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    score = float(
        similarity * 100
    )

    return max(
        0.0,
        min(
            100.0,
            score,
        ),
    )


# ============================================================
# Sentence splitting
# ============================================================

def split_into_sentences(
    text: str
) -> List[str]:

    cleaned = re.sub(
        r"\s+",
        " ",
        text.strip(),
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(
            sentence.strip()
        ) > 3
    ]


# ============================================================
# Requirement level
# ============================================================

def classify_requirement_level(
    sentence: str
) -> str:

    lower = sentence.lower()

    for phrase in OPTIONAL_INDICATORS:

        if phrase in lower:
            return "optional"

    for phrase in PREFERRED_INDICATORS:

        if phrase in lower:
            return "preferred"

    for phrase in REQUIRED_INDICATORS:

        if phrase in lower:
            return "required"

    return "required"


# ============================================================
# Alternative requirements
# ============================================================

def detect_alternative_skill_group(
    sentence: str,
    skills: List[str],
) -> Optional[List[str]]:

    if len(skills) < 2:
        return None

    lower = sentence.lower()

    has_alternative_language = any(
        marker in lower
        for marker in [
            " or ",
            "either ",
            "one of",
            "any of",
        ]
    )

    if not has_alternative_language:
        return None

    # --------------------------------------------------------
    # Cloud group
    # --------------------------------------------------------

    cloud_skills = {
        "aws",
        "azure",
        "gcp",
    }

    detected_cloud = [
        skill
        for skill in skills
        if skill in cloud_skills
    ]

    if len(detected_cloud) >= 2:
        return detected_cloud

    # --------------------------------------------------------
    # General OR group
    # --------------------------------------------------------

    if " or " in lower:
        return skills

    return None


# ============================================================
# Job requirement construction
# ============================================================

def build_job_requirements(
    job_description: str
) -> List[dict]:

    sentences = split_into_sentences(
        job_description
    )

    requirements = []

    seen_single_skills = set()

    for sentence in sentences:

        skills = extract_normalized_skills(
            sentence
        )

        if not skills:
            continue

        level = classify_requirement_level(
            sentence
        )

        alternative_group = (
            detect_alternative_skill_group(
                sentence,
                skills,
            )
        )

        # ----------------------------------------------------
        # Alternative group
        # ----------------------------------------------------

        if alternative_group:

            requirements.append(
                {
                    "type": "alternative",

                    "skills": sorted(
                        alternative_group
                    ),

                    "level":
                        level,

                    "weight":
                        REQUIREMENT_WEIGHTS[
                            level
                    ],

                    "source_text":
                        sentence,
                }
            )

            for skill in alternative_group:

                seen_single_skills.add(
                    skill
                )

            remaining_skills = [
                skill
                for skill in skills
                if skill not in alternative_group
            ]

            for skill in remaining_skills:

                if skill in seen_single_skills:
                    continue

                requirements.append(
                    {
                        "type": "single",

                        "skill":
                            skill,

                        "level":
                            level,

                        "weight":
                            REQUIREMENT_WEIGHTS[
                                level
                            ],

                        "source_text":
                            sentence,
                    }
                )

                seen_single_skills.add(
                    skill
                )

            continue

        # ----------------------------------------------------
        # Standard requirements
        # ----------------------------------------------------

        for skill in skills:

            if skill in seen_single_skills:
                continue

            requirements.append(
                {
                    "type": "single",

                    "skill":
                        skill,

                    "level":
                        level,

                    "weight":
                        REQUIREMENT_WEIGHTS[
                            level
                        ],

                    "source_text":
                        sentence,
                }
            )

            seen_single_skills.add(
                skill
            )

    return requirements


# ============================================================
# Weighted requirement scoring
# ============================================================

def evaluate_job_requirements(
    resume_skills: List[str],
    job_requirements: List[dict],
) -> dict:

    resume_skill_set = set(
        resume_skills
    )

    total_weight = 0.0
    matched_weight = 0.0

    matched_required = []
    missing_required = []

    matched_preferred = []
    missing_preferred = []

    matched_optional = []
    missing_optional = []

    alternative_requirements = []

    for requirement in job_requirements:

        weight = requirement[
            "weight"
        ]

        level = requirement[
            "level"
        ]

        total_weight += weight

        # ----------------------------------------------------
        # Alternative requirement
        # ----------------------------------------------------

        if requirement["type"] == "alternative":

            skills = requirement[
                "skills"
            ]

            matched_alternatives = [
                skill
                for skill in skills
                if skill in resume_skill_set
            ]

            satisfied = bool(
                matched_alternatives
            )

            if satisfied:
                matched_weight += weight

            alternative_requirements.append(
                {
                    "skills":
                        skills,

                    "level":
                        level,

                    "matched":
                        matched_alternatives,

                    "satisfied":
                        satisfied,

                    "source_text":
                        requirement[
                            "source_text"
                        ],
                }
            )

            continue

        # ----------------------------------------------------
        # Single skill
        # ----------------------------------------------------

        skill = requirement[
            "skill"
        ]

        matched = (
            skill in resume_skill_set
        )

        if matched:
            matched_weight += weight

        if level == "required":

            if matched:
                matched_required.append(
                    skill
                )
            else:
                missing_required.append(
                    skill
                )

        elif level == "preferred":

            if matched:
                matched_preferred.append(
                    skill
                )
            else:
                missing_preferred.append(
                    skill
                )

        else:

            if matched:
                matched_optional.append(
                    skill
                )
            else:
                missing_optional.append(
                    skill
                )

    if total_weight == 0:

        score = None

    else:

        score = (
            matched_weight
            / total_weight
        ) * 100

    return {

        "weighted_skill_score": (
            round(
                score,
                2,
            )
            if score is not None
            else None
        ),

        "required": {

            "matched": sorted(
                set(
                    matched_required
                )
            ),

            "missing": sorted(
                set(
                    missing_required
                )
            ),
        },

        "preferred": {

            "matched": sorted(
                set(
                    matched_preferred
                )
            ),

            "missing": sorted(
                set(
                    missing_preferred
                )
            ),
        },

        "optional": {

            "matched": sorted(
                set(
                    matched_optional
                )
            ),

            "missing": sorted(
                set(
                    missing_optional
                )
            ),
        },

        "alternative_requirements":
            alternative_requirements,
    }


# ============================================================
# Section semantic scoring
# ============================================================

def calculate_section_scores(
    resume_sections: Dict[str, str],
    job_description: str,
) -> Dict[str, float]:

    scores = {}

    for section_name, section_text in resume_sections.items():

        if section_name == "general":
            continue

        if len(
            section_text.strip()
        ) < 20:
            continue

        score = calculate_semantic_similarity(
            section_text,
            job_description,
        )

        scores[
            section_name
        ] = round(
            score,
            2,
        )

    return scores


# ============================================================
# Section evidence score
# ============================================================

def calculate_section_evidence_score(
    section_scores: Dict[str, float]
) -> Optional[float]:

    important_sections = [
        "experience",
        "projects",
        "skills",
        "summary",
    ]

    available_scores = [
        section_scores[
            section
        ]
        for section in important_sections
        if section in section_scores
    ]

    if not available_scores:
        return None

    top_scores = sorted(
        available_scores,
        reverse=True,
    )[:2]

    return (
        sum(top_scores)
        / len(top_scores)
    )


# ============================================================
# Resume chunking
# ============================================================

def create_resume_chunks(
    resume_sections: Dict[str, str],
    max_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[dict]:

    chunks = []

    for section_name, section_text in resume_sections.items():

        words = section_text.split()

        if not words:
            continue

        start = 0
        chunk_index = 0

        while start < len(words):

            end = min(
                start + max_words,
                len(words),
            )

            chunk_text = " ".join(
                words[
                    start:end
                ]
            ).strip()

            if chunk_text:

                chunks.append(
                    {
                        "section":
                            section_name,

                        "chunk_index":
                            chunk_index,

                        "text":
                            chunk_text,
                    }
                )

            if end >= len(words):
                break

            start = max(
                end - overlap_words,
                start + 1,
            )

            chunk_index += 1

    return chunks


# ============================================================
# Build skill-specific semantic query
# ============================================================

def build_requirement_evidence_query(
    requirement: dict
) -> str:

    level = requirement[
        "level"
    ]

    if requirement[
        "type"
    ] == "alternative":

        skills = requirement[
            "skills"
        ]

        skill_text = " or ".join(
            skills
        )

        return (
            f"professional experience or project experience "
            f"using {skill_text}"
        )

    skill = requirement[
        "skill"
    ]

    # --------------------------------------------------------
    # Tailored query wording
    # --------------------------------------------------------

    query_templates = {

        "machine learning":
            "machine learning project experience building and evaluating models",

        "feature engineering":
            "feature engineering experience creating transforming and selecting features",

        "classification":
            "classification machine learning project using classification models",

        "regression":
            "regression machine learning project using regression models",

        "cross-validation":
            "cross-validation model validation and machine learning evaluation experience",

        "model evaluation":
            "machine learning model evaluation metrics validation and performance analysis",

        "fastapi":
            "FastAPI backend development building REST APIs for machine learning applications",

        "cloud deployment":
            "deploying applications or machine learning services to cloud platforms",

        "postgresql":
            "PostgreSQL database development and application integration",

        "problem solving":
            "problem solving experience delivering technical projects and solving real problems",

        "nlp":
            "natural language processing NLP project experience",

        "transformers":
            "transformer models NLP project experience",

        "embeddings":
            "sentence embeddings vector embeddings machine learning NLP experience",

        "semantic search":
            "semantic search embedding based information retrieval project",

        "hugging face":
            "Hugging Face transformer model development experience",

        "python":
            "Python programming experience in machine learning projects",

        "pandas":
            "Pandas data analysis and machine learning data preparation experience",

        "numpy":
            "NumPy numerical computing and machine learning experience",

        "scikit-learn":
            "scikit-learn machine learning model development experience",

        "docker":
            "Docker containerization and application deployment experience",

        "git":
            "Git version control software development experience",

        "github":
            "GitHub software development and project collaboration experience",

        "sql":
            "SQL database query and application development experience",
    }

    if skill in query_templates:

        return query_templates[
            skill
        ]

    if level == "required":

        return (
            f"professional experience or project experience "
            f"demonstrating {skill}"
        )

    return (
        f"knowledge or project experience using {skill}"
    )


# ============================================================
# Explicit skill match in chunk
# ============================================================

def chunk_contains_skill(
    chunk_text: str,
    requirement: dict,
) -> bool:

    chunk_skills = set(
        extract_normalized_skills(
            chunk_text
        )
    )

    if requirement[
        "type"
    ] == "alternative":

        return any(
            skill in chunk_skills
            for skill in requirement[
                "skills"
            ]
        )

    return (
        requirement[
            "skill"
        ]
        in chunk_skills
    )


# ============================================================
# Evidence strength
# ============================================================

def classify_evidence_strength(
    score: float,
    explicit_skill_match: bool,
) -> str:

    # Exact skill mention is strong evidence that the
    # terminology exists in the resume. Semantic evidence
    # still gives context around that mention.

    if explicit_skill_match and score >= 45:
        return "strong"

    if score >= 70:
        return "strong"

    if score >= 55:
        return "moderate"

    if score >= 40:
        return "weak"

    return "very_weak"


# ============================================================
# Requirement semantic evidence retrieval
# ============================================================

def retrieve_requirement_evidence(
    job_requirements: List[dict],
    resume_chunks: List[dict],
    resume_skills: List[str],
    top_k: int = TOP_EVIDENCE_RESULTS,
) -> List[dict]:

    if (
        not job_requirements
        or not resume_chunks
    ):
        return []

    # --------------------------------------------------------
    # Embed resume chunks once
    # --------------------------------------------------------

    resume_texts = [
        chunk[
            "text"
        ]
        for chunk in resume_chunks
    ]

    resume_embeddings = encode_texts(
        resume_texts
    )

    # --------------------------------------------------------
    # Create requirement-specific search queries
    # --------------------------------------------------------

    evidence_queries = [
        build_requirement_evidence_query(
            requirement
        )
        for requirement in job_requirements
    ]

    query_embeddings = encode_texts(
        evidence_queries
    )

    # --------------------------------------------------------
    # Similarity matrix
    # --------------------------------------------------------

    similarity_matrix = cosine_similarity(
        query_embeddings,
        resume_embeddings,
    )

    resume_skill_set = set(
        resume_skills
    )

    evidence_results = []

    # --------------------------------------------------------
    # Each requirement
    # --------------------------------------------------------

    for requirement_index, requirement in enumerate(
        job_requirements
    ):

        similarities = similarity_matrix[
            requirement_index
        ]

        ranked_indexes = (
            similarities
            .argsort()[::-1]
        )

        top_matches = []

        for resume_index in ranked_indexes[
            :top_k
        ]:

            score = float(
                similarities[
                    resume_index
                ] * 100
            )

            score = max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            )

            chunk = resume_chunks[
                resume_index
            ]

            explicit_chunk_match = (
                chunk_contains_skill(
                    chunk[
                        "text"
                    ],
                    requirement,
                )
            )

            evidence_strength = (
                classify_evidence_strength(
                    score,
                    explicit_chunk_match,
                )
            )

            top_matches.append(
                {
                    "section":
                        chunk[
                            "section"
                        ],

                    "chunk_index":
                        chunk[
                            "chunk_index"
                        ],

                    "text":
                        chunk[
                            "text"
                        ],

                    "score":
                        round(
                            score,
                            2,
                        ),

                    "explicit_skill_match":
                        explicit_chunk_match,

                    "evidence_strength":
                        evidence_strength,
                }
            )

        # ----------------------------------------------------
        # Requirement-level explicit match
        # ----------------------------------------------------

        if requirement[
            "type"
        ] == "alternative":

            requirement_name = " OR ".join(
                requirement[
                    "skills"
                ]
            )

            explicitly_present = any(
                skill in resume_skill_set
                for skill in requirement[
                    "skills"
                ]
            )

        else:

            requirement_name = requirement[
                "skill"
            ]

            explicitly_present = (
                requirement_name
                in resume_skill_set
            )

        best_score = (
            top_matches[0][
                "score"
            ]
            if top_matches
            else None
        )

        best_strength = (
            top_matches[0][
                "evidence_strength"
            ]
            if top_matches
            else "none"
        )

        evidence_results.append(
            {
                "requirement":
                    requirement_name,

                "requirement_type":
                    requirement[
                        "type"
                    ],

                "level":
                    requirement[
                        "level"
                    ],

                "source_text":
                    requirement[
                        "source_text"
                    ],

                "semantic_query":
                    evidence_queries[
                        requirement_index
                    ],

                "explicit_skill_match":
                    explicitly_present,

                "best_evidence_score":
                    best_score,

                "evidence_strength":
                    best_strength,

                "evidence":
                    top_matches,
            }
        )

    return evidence_results


# ============================================================
# Match classification
# ============================================================

def classify_match(
    overall_score: float
) -> str:

    if overall_score >= 80:
        return "Strong Match"

    if overall_score >= 65:
        return "Good Match"

    if overall_score >= 50:
        return "Partial Match"

    return "Weak Match"


# ============================================================
# Recommendation
# ============================================================

def generate_recommendation(
    match_level: str,
    requirement_analysis: dict,
) -> str:

    missing_required = (
        requirement_analysis[
            "required"
        ][
            "missing"
        ]
    )

    missing_preferred = (
        requirement_analysis[
            "preferred"
        ][
            "missing"
        ]
    )

    if missing_required:

        top_missing = ", ".join(
            missing_required[:4]
        )

        return (
            "The resume has useful alignment with this role, "
            "but some core requirements are not clearly demonstrated: "
            f"{top_missing}. Review the semantic evidence before treating "
            "these as true skill gaps. Only add skills to the resume when "
            "they accurately reflect real knowledge, projects, or experience."
        )

    if missing_preferred:

        top_missing = ", ".join(
            missing_preferred[:4]
        )

        return (
            "The core requirements are well aligned. "
            "Preferred skills that could further strengthen the profile "
            f"include: {top_missing}. Only add them when they reflect "
            "genuine experience or knowledge."
        )

    if match_level == "Strong Match":

        return (
            "Strong alignment with this role. Emphasize measurable project "
            "impact, production deployment experience, and the strongest "
            "resume evidence supporting the job requirements."
        )

    return (
        "The resume demonstrates useful alignment with this role. "
        "Strengthen the application by highlighting measurable outcomes "
        "and concrete project evidence for the most important requirements."
    )


# ============================================================
# Main evaluation
# ============================================================

def evaluate_match(
    resume_text: str,
    job_description: str,
) -> dict:

    # --------------------------------------------------------
    # Full resume ↔ JD semantic score
    # --------------------------------------------------------

    semantic_score = (
        calculate_semantic_similarity(
            resume_text,
            job_description,
        )
    )

    # --------------------------------------------------------
    # Skill extraction
    # --------------------------------------------------------

    resume_skills = (
        extract_normalized_skills(
            resume_text
        )
    )

    job_skills = (
        extract_normalized_skills(
            job_description
        )
    )

    # --------------------------------------------------------
    # Job requirements
    # --------------------------------------------------------

    job_requirements = (
        build_job_requirements(
            job_description
        )
    )

    requirement_analysis = (
        evaluate_job_requirements(
            resume_skills,
            job_requirements,
        )
    )

    weighted_skill_score = (
        requirement_analysis[
            "weighted_skill_score"
        ]
    )

    # --------------------------------------------------------
    # Frontend-compatible skill fields
    # --------------------------------------------------------

    matched_skills = sorted(
        set(
            resume_skills
        )
        & set(
            job_skills
        )
    )

    missing_skills = sorted(
        set(
            job_skills
        )
        - set(
            resume_skills
        )
    )

    skill_analysis_available = bool(
        job_requirements
    )

    # --------------------------------------------------------
    # Resume sections
    # --------------------------------------------------------

    resume_sections = (
        detect_resume_sections(
            resume_text
        )
    )

    # --------------------------------------------------------
    # Resume chunks
    # --------------------------------------------------------

    resume_chunks = (
        create_resume_chunks(
            resume_sections
        )
    )

    # --------------------------------------------------------
    # Semantic requirement evidence
    # --------------------------------------------------------

    requirement_evidence = (
        retrieve_requirement_evidence(
            job_requirements,
            resume_chunks,
            resume_skills,
            top_k=TOP_EVIDENCE_RESULTS,
        )
    )

    # --------------------------------------------------------
    # Section-level scores
    # --------------------------------------------------------

    section_scores = (
        calculate_section_scores(
            resume_sections,
            job_description,
        )
    )

    section_evidence_score = (
        calculate_section_evidence_score(
            section_scores
        )
    )

    # --------------------------------------------------------
    # Overall scoring
    #
    # IMPORTANT:
    # Requirement evidence is NOT yet included here.
    # We avoid double-counting until evidence thresholds
    # have been properly evaluated.
    # --------------------------------------------------------

    if (
        weighted_skill_score is not None
        and section_evidence_score is not None
    ):

        overall_score = (
            semantic_score * 0.45
            + weighted_skill_score * 0.35
            + section_evidence_score * 0.20
        )

        scoring_method = (
            "semantic + weighted requirements "
            "+ section evidence"
        )

    elif weighted_skill_score is not None:

        overall_score = (
            semantic_score * 0.60
            + weighted_skill_score * 0.40
        )

        scoring_method = (
            "semantic + weighted requirements"
        )

    elif section_evidence_score is not None:

        overall_score = (
            semantic_score * 0.75
            + section_evidence_score * 0.25
        )

        scoring_method = (
            "semantic + section evidence"
        )

    else:

        overall_score = (
            semantic_score
        )

        scoring_method = (
            "semantic only"
        )

    # --------------------------------------------------------
    # Match level
    # --------------------------------------------------------

    match_level = (
        classify_match(
            overall_score
        )
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    recommendation = (
        generate_recommendation(
            match_level,
            requirement_analysis,
        )
    )

    # --------------------------------------------------------
    # Best matching sections
    # --------------------------------------------------------

    best_matching_sections = sorted(
        section_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    best_matching_sections = [
        {
            "section":
                section,

            "score":
                score,
        }
        for section, score
        in best_matching_sections
    ]

    # --------------------------------------------------------
    # Evidence summary
    # --------------------------------------------------------

    strong_evidence_count = sum(
        1
        for item in requirement_evidence
        if item[
            "evidence_strength"
        ] == "strong"
    )

    moderate_evidence_count = sum(
        1
        for item in requirement_evidence
        if item[
            "evidence_strength"
        ] == "moderate"
    )

    weak_evidence_count = sum(
        1
        for item in requirement_evidence
        if item[
            "evidence_strength"
        ] == "weak"
    )

    very_weak_evidence_count = sum(
        1
        for item in requirement_evidence
        if item[
            "evidence_strength"
        ] == "very_weak"
    )

    # --------------------------------------------------------
    # Final API response
    # --------------------------------------------------------

    return {

        "semantic_score":
            round(
                semantic_score,
                2,
            ),

        # Backwards compatibility

        "skill_match_score":
            weighted_skill_score,

        "weighted_skill_score":
            weighted_skill_score,

        "overall_score":
            round(
                float(
                    overall_score
                ),
                2,
            ),

        "match_level":
            match_level,

        "matched_skills":
            matched_skills,

        "missing_skills":
            missing_skills,

        "skill_analysis_available":
            skill_analysis_available,

        "recommendation":
            recommendation,

        # ----------------------------------------------------
        # Structured requirements
        # ----------------------------------------------------

        "requirement_analysis":
            requirement_analysis,

        "job_requirements":
            job_requirements,

        # ----------------------------------------------------
        # Semantic evidence
        # ----------------------------------------------------

        "requirement_evidence":
            requirement_evidence,

        "evidence_summary": {

            "strong":
                strong_evidence_count,

            "moderate":
                moderate_evidence_count,

            "weak":
                weak_evidence_count,

            "very_weak":
                very_weak_evidence_count,
                },

        "resume_chunk_count":
            len(
                resume_chunks
        ),

        "semantic_search_enabled":
            True,

        # ----------------------------------------------------
        # Resume sections
        # ----------------------------------------------------

        "section_evidence_score": (
            round(
                section_evidence_score,
                2,
            )
            if section_evidence_score
            is not None
            else None
                ),

        "section_scores":
            section_scores,

        "best_matching_sections":
            best_matching_sections,

        "resume_sections_detected":
            sorted(
                resume_sections.keys()
        ),

        # ----------------------------------------------------
        # Skills
        # ----------------------------------------------------

        "resume_skills_detected":
            resume_skills,

        "job_skills_detected":
            job_skills,

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "scoring_method":
            scoring_method,

        "model_name":
            MODEL_NAME,

        "api_version":
            API_VERSION,
    }


# ============================================================
# Root
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "AI Resume Job Match API is running",

        "version":
            API_VERSION,

        "features": [
            "transformer semantic matching",
            "weighted job requirement analysis",
            "required/preferred/optional detection",
            "alternative OR requirement handling",
            "skill-gap detection",
            "resume section detection",
            "chunk-based semantic retrieval",
            "skill-specific evidence search",
            "explicit skill evidence detection",
            "evidence confidence classification",
        ],
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model":
            MODEL_NAME,

        "version":
            API_VERSION,

        "semantic_search":
            True,
    }


# ============================================================
# Analyze endpoint
# ============================================================

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):

    # --------------------------------------------------------
    # Validate PDF
    # --------------------------------------------------------

    if resume.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a PDF resume."
            ),
        )

    # --------------------------------------------------------
    # Validate JD
    # --------------------------------------------------------

    if not job_description.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Job description cannot be empty."
            ),
        )

    # --------------------------------------------------------
    # Read resume
    # --------------------------------------------------------

    file_bytes = await resume.read()

    if not file_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded resume is empty."
            ),
        )

    if len(
        file_bytes
    ) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Resume PDF is too large. "
                "Maximum allowed size is 5 MB."
            ),
        )

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    try:

        resume_text = (
            extract_pdf_text(
                file_bytes
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc

    if not resume_text.strip():

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract readable "
                "text from the resume PDF."
            ),
        )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    logger.info(
        "Running Resume–Job analysis."
    )

    result = evaluate_match(
        resume_text=resume_text,
        job_description=job_description.strip(),
    )

    return result
