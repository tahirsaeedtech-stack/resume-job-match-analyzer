import { useState } from "react";
import "./App.css";

const API_BASE_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAllEvidence, setShowAllEvidence] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!resumeFile || !jobDescription.trim()) {
      setError(
        "Please upload a resume PDF and enter a job description."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setShowAllEvidence(false);

    const formData = new FormData();
    formData.append("resume", resumeFile);
    formData.append("job_description", jobDescription.trim());

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      let data;

      try {
        data = await response.json();
      } catch {
        data = {};
      }

      if (!response.ok) {
        throw new Error(
          data.detail ||
          data.error ||
          `Analysis failed with status ${response.status}.`
        );
      }

      setResult(data);
    } catch (err) {
      console.error("Resume analysis request failed:", err);

      setError(
        err.message ||
        "Could not connect to the analysis backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const getMatchClass = (level) => {
    if (level === "Strong Match") return "strong";
    if (level === "Good Match") return "good";
    if (level === "Partial Match") return "moderate";
    return "weak";
  };

  const getEvidenceClass = (strength) => {
    if (strength === "strong") return "strong";
    if (strength === "moderate") return "moderate";
    if (strength === "weak") return "weak";
    return "very-weak";
  };

  const getEvidenceLabel = (strength) => {
    if (strength === "strong") return "Strong Evidence";
    if (strength === "moderate") return "Moderate Evidence";
    if (strength === "weak") return "Limited Evidence";
    return "No Clear Evidence";
  };

  const formatSkill = (skill) => {
    if (!skill) return "";

    return skill
      .split(" ")
      .map((word) =>
        word.length > 2
          ? word.charAt(0).toUpperCase() + word.slice(1)
          : word.toUpperCase()
      )
      .join(" ");
  };

  const formatSection = (section) => {
    if (!section) return "";

    return section
      .replaceAll("_", " ")
      .split(" ")
      .map(
        (word) =>
          word.charAt(0).toUpperCase() + word.slice(1)
      )
      .join(" ");
  };

  const truncateText = (text, maxLength = 220) => {
    if (!text) return "";

    if (text.length <= maxLength) {
      return text;
    }

    return `${text.slice(0, maxLength).trim()}...`;
  };

  const renderSkillTags = (skills, type = "matched") => {
    if (!skills || skills.length === 0) {
      return <p className="muted-text">None detected.</p>;
    }

    return (
      <div className="tags">
        {skills.map((skill) => (
          <span
            className={`tag ${type}`}
            key={skill}
          >
            {formatSkill(skill)}
          </span>
        ))}
      </div>
    );
  };

  const renderRequirementGroup = (
    title,
    matched = [],
    missing = []
  ) => {
    return (
      <div className="requirement-group">
        <h4>{title}</h4>

        {matched.length > 0 && (
          <>
            <p className="sub-label">Matched</p>

            <div className="tags">
              {matched.map((skill) => (
                <span
                  className="tag matched"
                  key={`matched-${skill}`}
                >
                  ✓ {formatSkill(skill)}
                </span>
              ))}
            </div>
          </>
        )}

        {missing.length > 0 && (
          <>
            <p className="sub-label">
              Not clearly demonstrated
            </p>

            <div className="tags">
              {missing.map((skill) => (
                <span
                  className="tag missing"
                  key={`missing-${skill}`}
                >
                  ○ {formatSkill(skill)}
                </span>
              ))}
            </div>
          </>
        )}

        {matched.length === 0 &&
          missing.length === 0 && (
            <p className="muted-text">
              No recognized requirements in this group.
            </p>
          )}
      </div>
    );
  };

  const renderAlternativeRequirements = () => {
    const alternatives =
      result?.requirement_analysis
        ?.alternative_requirements || [];

    if (alternatives.length === 0) {
      return null;
    }

    return (
      <div className="requirement-group">
        <h4>Alternative Requirements</h4>

        {alternatives.map((requirement, index) => (
          <div
            className="alternative-card"
            key={`${requirement.skills.join("-")}-${index}`}
          >
            <p className="alternative-title">
              {requirement.skills
                .map(formatSkill)
                .join(" OR ")}
            </p>

            <p
              className={
                requirement.satisfied
                  ? "alternative-status satisfied"
                  : "alternative-status unsatisfied"
              }
            >
              {requirement.satisfied
                ? `Satisfied by: ${(
                  requirement.matched || []
                )
                  .map(formatSkill)
                  .join(", ")}`
                : "Not demonstrated"}
            </p>

            <span className="requirement-level">
              {formatSection(requirement.level)}
            </span>
          </div>
        ))}
      </div>
    );
  };

  const getEvidencePriority = (item) => {
    const requiredMissing =
      result?.requirement_analysis?.required?.missing || [];

    const preferredMissing =
      result?.requirement_analysis?.preferred?.missing || [];

    if (
      item.level === "required" &&
      requiredMissing.includes(item.requirement)
    ) {
      return 0;
    }

    if (
      item.level === "preferred" &&
      preferredMissing.includes(item.requirement)
    ) {
      return 1;
    }

    if (item.level === "required") {
      return 2;
    }

    if (item.level === "preferred") {
      return 3;
    }

    return 4;
  };

  const prioritizedEvidence =
    result?.requirement_evidence
      ?.filter(
        (item) =>
          item.level === "required" ||
          item.level === "preferred"
      )
      .slice()
      .sort((a, b) => {
        const priorityDifference =
          getEvidencePriority(a) -
          getEvidencePriority(b);

        if (priorityDifference !== 0) {
          return priorityDifference;
        }

        return (
          (b.best_evidence_score || 0) -
          (a.best_evidence_score || 0)
        );
      }) || [];

  const visibleEvidence = showAllEvidence
    ? prioritizedEvidence
    : prioritizedEvidence.slice(0, 4);

  return (
    <div className="app">
      <section className="hero">
        <div>
          <span className="eyebrow">
            AI Resume Intelligence
          </span>

          <h1>Resume–Job Match Analyzer</h1>

          <p>
            Analyze candidate-job alignment using
            transformer embeddings, weighted job
            requirements, skill-gap detection, and resume
            evidence retrieval.
          </p>
        </div>
      </section>

      <main className="container">
        <div className="grid">
          <section className="card">
            <h2>Analyze Candidate Fit</h2>

            <form onSubmit={handleSubmit}>
              <div className="field">
                <label htmlFor="resume">
                  Resume PDF
                </label>

                <input
                  id="resume"
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={(e) => {
                    setResumeFile(
                      e.target.files?.[0] || null
                    );
                    setError("");
                  }}
                />

                {resumeFile && (
                  <p className="file-name">
                    Selected: {resumeFile.name}
                  </p>
                )}
              </div>

              <div className="field">
                <label htmlFor="job-description">
                  Job Description
                </label>

                <textarea
                  id="job-description"
                  rows="14"
                  placeholder="Paste the job description here..."
                  value={jobDescription}
                  onChange={(e) => {
                    setJobDescription(
                      e.target.value
                    );
                    setError("");
                  }}
                />
              </div>

              <button
                type="submit"
                disabled={loading}
              >
                {loading
                  ? "Analyzing..."
                  : "Analyze Match"}
              </button>
            </form>

            {error && (
              <div className="error">{error}</div>
            )}
          </section>

          <section className="card result-card">
            {!result ? (
              <div className="empty-state">
                <h2>Match Analysis</h2>

                <p>
                  Upload a resume and job description to
                  view semantic alignment, requirement
                  coverage, skill gaps, and supporting
                  resume evidence.
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
                      {Number(
                        result.overall_score
                      ).toFixed(1)}
                      %
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
                    <span>Semantic Alignment</span>

                    <strong>
                      {Number(
                        result.semantic_score
                      ).toFixed(1)}
                      %
                    </strong>
                  </div>

                  <div className="metric">
                    <span>Requirement Match</span>

                    <strong>
                      {result.weighted_skill_score !==
                        null
                        ? `${Number(
                          result.weighted_skill_score
                        ).toFixed(1)}%`
                        : "N/A"}
                    </strong>
                  </div>

                  <div className="metric">
                    <span>Section Evidence</span>

                    <strong>
                      {result.section_evidence_score !==
                        null
                        ? `${Number(
                          result.section_evidence_score
                        ).toFixed(1)}%`
                        : "N/A"}
                    </strong>
                  </div>
                </div>

                {result.requirement_analysis && (
                  <section className="analysis-section">
                    <div className="section-heading">
                      <div>
                        <p className="section-kicker">
                          Requirement Analysis
                        </p>

                        <h3>
                          Job Requirement Coverage
                        </h3>
                      </div>
                    </div>

                    {renderRequirementGroup(
                      "Core Requirements",
                      result.requirement_analysis
                        .required?.matched || [],
                      result.requirement_analysis
                        .required?.missing || []
                    )}

                    {renderRequirementGroup(
                      "Preferred Skills",
                      result.requirement_analysis
                        .preferred?.matched || [],
                      result.requirement_analysis
                        .preferred?.missing || []
                    )}

                    {renderAlternativeRequirements()}
                  </section>
                )}

                {prioritizedEvidence.length > 0 && (
                  <section className="analysis-section">
                    <div className="section-heading evidence-heading">
                      <div>
                        <p className="section-kicker">
                          Semantic Search
                        </p>

                        <h3>Resume Evidence</h3>
                      </div>
                    </div>

                    {result.evidence_summary && (
                      <div className="evidence-summary">
                        <span>
                          {
                            result.evidence_summary
                              .strong
                          }{" "}
                          strong
                        </span>

                        <span>
                          {
                            result.evidence_summary
                              .moderate
                          }{" "}
                          moderate
                        </span>

                        <span>
                          {
                            result.evidence_summary
                              .weak
                          }{" "}
                          limited
                        </span>

                        <span>
                          {
                            result.evidence_summary
                              .very_weak
                          }{" "}
                          unclear
                        </span>
                      </div>
                    )}

                    <div className="evidence-list">
                      {visibleEvidence.map((item) => {
                        const evidence =
                          item.evidence?.[0];

                        return (
                          <article
                            className="evidence-card"
                            key={`${item.requirement}-${item.level}`}
                          >
                            <div className="evidence-top">
                              <div>
                                <h4>
                                  {formatSkill(
                                    item.requirement
                                  )}
                                </h4>

                                <p className="evidence-meta">
                                  {formatSection(
                                    item.level
                                  )}{" "}
                                  requirement
                                </p>
                              </div>

                              <span
                                className={`evidence-badge ${getEvidenceClass(
                                  item.evidence_strength
                                )}`}
                              >
                                {getEvidenceLabel(
                                  item.evidence_strength
                                )}
                              </span>
                            </div>

                            <div className="evidence-status-row">
                              <span
                                className={
                                  item.explicit_skill_match
                                    ? "explicit-status yes"
                                    : "explicit-status no"
                                }
                              >
                                {item.explicit_skill_match
                                  ? "✓ Explicit skill mention"
                                  : "○ No explicit skill mention"}
                              </span>

                              {item.best_evidence_score !==
                                null &&
                                item.best_evidence_score !==
                                undefined && (
                                  <span className="evidence-score">
                                    Semantic evidence:{" "}
                                    {Number(
                                      item.best_evidence_score
                                    ).toFixed(1)}
                                    %
                                  </span>
                                )}
                            </div>

                            {evidence ? (
                              <div className="evidence-content">
                                <p className="evidence-section-name">
                                  Best evidence from{" "}
                                  <strong>
                                    {formatSection(
                                      evidence.section
                                    )}
                                  </strong>
                                </p>

                                <p className="evidence-text">
                                  “
                                  {truncateText(
                                    evidence.text
                                  )}
                                  ”
                                </p>
                              </div>
                            ) : (
                              <p className="muted-text">
                                No supporting resume
                                evidence was retrieved.
                              </p>
                            )}
                          </article>
                        );
                      })}
                    </div>

                    {prioritizedEvidence.length > 4 && (
                      <button
                        type="button"
                        className="secondary-button evidence-toggle"
                        onClick={() =>
                          setShowAllEvidence(
                            (previous) => !previous
                          )
                        }
                      >
                        {showAllEvidence
                          ? "Show Less Evidence"
                          : `Show All Evidence (${prioritizedEvidence.length})`}
                      </button>
                    )}
                  </section>
                )}

                <section className="analysis-section">
                  <div className="section-heading">
                    <div>
                      <p className="section-kicker">
                        Skills
                      </p>

                      <h3>
                        Explicitly Matched Skills
                      </h3>
                    </div>
                  </div>

                  {renderSkillTags(
                    result.matched_skills,
                    "matched"
                  )}
                </section>

                {result.resume_sections_detected
                  ?.length > 0 && (
                    <section className="analysis-section">
                      <div className="section-heading">
                        <div>
                          <p className="section-kicker">
                            Resume Structure
                          </p>

                          <h3>Detected Sections</h3>
                        </div>
                      </div>

                      <div className="tags">
                        {result.resume_sections_detected.map(
                          (section) => (
                            <span
                              className="tag neutral"
                              key={section}
                            >
                              {formatSection(section)}
                            </span>
                          )
                        )}
                      </div>
                    </section>
                  )}

                <div className="recommendation">
                  <h3>Recommendation</h3>

                  <p>{result.recommendation}</p>
                </div>

                <div className="note">
                  This analysis combines transformer
                  semantic similarity, weighted requirement
                  matching, and resume section evidence. It
                  is an explainable engineering heuristic,
                  not a hiring probability or an official
                  ATS score.
                </div>

                <div className="api-meta">
                  <span>
                    Model: {result.model_name}
                  </span>

                  <span>
                    API v{result.api_version}
                  </span>
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