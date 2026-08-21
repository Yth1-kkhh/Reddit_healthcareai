# Annotation schema

This document defines the labels produced by every agent and the input/output
JSON structures. All three models share the same prompt structure; only the
stakeholder role definition and the two in-context examples differ per agent.

## Per-thread decision sequence

Each agent makes the following judgements for a thread:

1. **Stakeholder present** — is the agent's target stakeholder group actively
   *self-presented* in the thread (first-person role evidence), rather than
   merely mentioned? Field name depends on the agent:
   - `healthcare_professionals` (HCP agents)
   - `software_engineers` (SEng agents)
   - `healthcare_researchers` (HCR agents)
2. `discusses_ai_in_healthcare` — does the thread concern AI/ML in a healthcare
   context?
3. `discusses_ethical_implications` — does it raise at least one ethics-related
   concern?
4. `ethical_dimensions` — a boolean for each of the five dimensions.
5. `themes` — for each dimension marked `true`, 3–7 short supporting phrases
   extracted from the thread; empty list otherwise.

## Ethical dimensions

| Dimension        | Marked `true` when the thread discusses ...                     |
|------------------|-----------------------------------------------------------------|
| `safety`         | patient safety, reliability, error, harm, clinical risk         |
| `privacy`        | data protection, confidentiality, consent to data use           |
| `bias`           | fairness, demographic/dataset bias, unequal performance         |
| `transparency`   | explainability, interpretability, disclosure, black-box concerns|
| `accountability` | responsibility, liability, oversight, auditability, sign-off    |

## Stakeholder inclusion criteria (summary)

- **Healthcare Professionals (HCP):** physicians, nurses, pharmacists, dentists,
  clinical-year medical students, licensed clinicians. Excludes patients,
  family, general public, and non-clinical health-IT roles.
- **Software Engineers (SEng):** ML engineers, developers, data scientists, AI
  researchers with active technical involvement ("As an ML engineer…", "When we
  deployed…", "Our training pipeline…"). Excludes passive tech users and generic
  "developers should…" statements.
- **Healthcare Researchers (HCR):** PhD students, postdocs, research scientists,
  PIs, IRB members with active research involvement ("In our clinical trial…",
  "Our IRB requires…", "As a PI…"). Excludes clinicians discussing routine
  practice and people merely citing studies.

## Output record

Each script writes a JSON list; one object per thread:

```json
{
  "post_id": "abc123",
  "subreddit": "medicine",
  "title": "...",
  "<model>_annotation": {
    "<stakeholder_field>": true,
    "discusses_ai_in_healthcare": true,
    "discusses_ethical_implications": true,
    "ethical_dimensions": {
      "safety": true, "privacy": false, "bias": false,
      "transparency": true, "accountability": true
    },
    "themes": {
      "safety": ["over-reliance on AI triage"],
      "privacy": [], "bias": [],
      "transparency": ["black box algorithm concerns"],
      "accountability": ["tool not replacement boundary"]
    }
  },
  "annotation_status": "success",
  "model_used": "gemini-2.5-flash-lite"
}
```

The annotation key is `gemini_annotation`, `grok_annotation`, or
`openai_annotation` depending on the model.

## Status codes

| `annotation_status`  | Meaning                                              |
|----------------------|------------------------------------------------------|
| `success`            | valid JSON returned and parsed                       |
| `json_error`         | output could not be parsed as JSON after retries     |
| `rate_limit_error`   | provider rate limit persisted after retries          |
| `error`              | other unrecoverable API error (message truncated)    |

## Aggregation (downstream)

A thread–stakeholder pair enters the final corpus when at least **two of the
three models** running that stakeholder agent agree the stakeholder is present,
**and** at least two of three agree on at least one ethical dimension. That
majority-vote aggregation is applied to the nine `results/*.json` files and is
not part of the annotation scripts in this repository.
