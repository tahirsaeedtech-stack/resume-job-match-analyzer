from backend.main import (
    app,
    extract_normalized_skills,
    classify_requirement_level,
    build_job_requirements,
)
from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert data["status"] == "healthy"


def test_analyze_rejects_non_pdf_file():
    fake_file = (
        "resume.txt",
        b"This is not a PDF.",
        "text/plain",
    )

    response = client.post(
        "/analyze",
        files={"resume": fake_file},
        data={
            "job_description": (
                "Looking for a Python machine learning engineer."
            )
        },
    )

    assert response.status_code in (400, 415, 422)


def test_analyze_rejects_empty_job_description():
    fake_pdf = (
        "resume.pdf",
        b"%PDF-1.4 fake pdf content",
        "application/pdf",
    )

    response = client.post(
        "/analyze",
        files={"resume": fake_pdf},
        data={"job_description": ""},
    )

    assert response.status_code in (400, 422)


def test_extract_normalized_skills_detects_common_ml_skills():
    text = """
    Python developer with experience in machine learning,
    pandas, NumPy, scikit-learn, FastAPI, Docker, Git and SQL.
    """

    skills = extract_normalized_skills(text)

    assert "python" in skills
    assert "machine learning" in skills
    assert "pandas" in skills
    assert "numpy" in skills
    assert "scikit-learn" in skills
    assert "fastapi" in skills
    assert "docker" in skills
    assert "git" in skills
    assert "sql" in skills


def test_requirement_classifier_detects_required_requirement():
    sentence = "The candidate must have experience with Python."

    level = classify_requirement_level(sentence)

    assert level == "required"


def test_requirement_classifier_detects_preferred_requirement():
    sentence = "Experience with FastAPI is preferred."

    level = classify_requirement_level(sentence)

    assert level == "preferred"


def test_alternative_cloud_requirement_is_grouped():
    job_description = """
    Experience with AWS, Azure, or GCP is beneficial.
    """

    requirements = build_job_requirements(job_description)

    alternative_requirements = [
        item
        for item in requirements
        if item.get("type") == "alternative"
    ]

    assert len(alternative_requirements) >= 1

    cloud_group = alternative_requirements[0]

    assert cloud_group["level"] == "optional"
    assert "aws" in cloud_group["skills"]
    assert "azure" in cloud_group["skills"]
    assert "gcp" in cloud_group["skills"]
