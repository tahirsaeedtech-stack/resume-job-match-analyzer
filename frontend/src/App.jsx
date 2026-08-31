import { useState } from "react";
import "./App.css";

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!resumeFile || !jobDescription.trim()) {
      setError("Please upload a resume PDF and enter a job description.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("job_description", jobDescription);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/analyze",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || "Analysis failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
        "Could not connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const getMatchClass = (level) => {
    if (level === "Strong Match") return "strong";
    if (level === "Moderate Match") return "moderate";
    return "weak";
  };

  return (
    <div className="app">
      <section className="hero">
        <div>
          <span className="eyebrow">AI Resume Intelligence</span>
          <h1>Resume–Job Match Analyzer</h1>
          <p>
            Compare a resume with a job description using semantic
            embeddings and skill-gap analysis.
          </p>
        </div>
      </section>

      <main className="container">
        <div className="grid">
          <section className="card">
            <h2>Analyze Candidate Fit</h2>

            <form onSubmit={handleSubmit}>
              <div className="field">
                <label>Resume PDF</label>
                <input
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={(e) =>
                    setResumeFile(e.target.files[0])
                  }
                />

                {resumeFile && (
                  <p className="file-name">
                    Selected: {resumeFile.name}
                  </p>
                )}
              </div>

              <div className="field">
                <label>Job Description</label>
                <textarea
                  rows="14"
                  placeholder="Paste the job description here..."
                  value={jobDescription}
                  onChange={(e) =>
                    setJobDescription(e.target.value)
                  }
                />
              </div>

              <button type="submit" disabled={loading}>
                {loading ? "Analyzing..." : "Analyze Match"}
              </button>
            </form>

            {error && <div className="error">{error}</div>}
          </section>

          <section className="card result-card">
            {!result ? (
              <div className="empty-state">
                <h2>Match Analysis</h2>
                <p>
                  Upload a resume and job description to view
                  semantic fit, skill coverage, and missing skills.
                </p>
              </div>
            ) : (
              <>
                <div className="result-header">
                  <div>
                    <p className="result-label">
                      Overall Match
                    </p>
                    <h2>
                      {result.overall_score.toFixed(2)}%
                    </h2>
                  </div>

                  <span
                    className={`badge ${getMatchClass(
                      result.match_level
                    )}`}
                  >
                    {result.match_level}
                  </span>
                </div>

                <div className="metrics">
                  <div className="metric">
                    <span>Semantic Score</span>
                    <strong>
                      {result.semantic_score.toFixed(2)}%
                    </strong>
                  </div>

                  <div className="metric">
                    <span>Skill Match</span>
                    <strong>
                      {result.skill_match_score !== null
                        ? `${result.skill_match_score.toFixed(2)}%`
                        : "N/A"}
                    </strong>
                  </div>
                </div>

                <div className="skills-section">
                  <h3>Missing Skills</h3>

                  <div className="tags">
                    {result.missing_skills.length > 0 ? (
                      result.missing_skills.map((skill) => (
                        <span
                          className="tag missing"
                          key={skill}
                        >
                          {skill}
                        </span>
                      ))
                    ) : (
                      <p>
                        {result.skill_analysis_available
                          ? "No major skill gaps detected."
                          : "Insufficient recognized skills for skill-gap analysis."}
                      </p>
                    )}
                  </div>
                </div>

                <div className="note">
                  {result.skill_analysis_available
                    ? "The score combines semantic similarity and explicit skill coverage. It is not a hiring probability."
                    : "The score is based primarily on semantic similarity because sufficient explicit job skills were not recognized. It is not a hiring probability."}
                </div>
                <div className="recommendation">
                  <h3>Recommendation</h3>
                  <p>{result.recommendation}</p>
                </div>
              </>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;