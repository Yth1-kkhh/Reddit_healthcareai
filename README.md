# Mining Stakeholder Perspectives for Healthcare AI — Annotation Replication Package

This repository contains the multi-agent LLM annotation pipeline used to identify
self-presented stakeholder participation and ethics-related discussion in Reddit
threads about healthcare AI.

Each Reddit thread is annotated **nine times**: three stakeholder-specific agents
(**Healthcare Professionals**, **Software Engineers**, **Healthcare Researchers**),
each run on three independently developed LLMs (**Gemini**, **Grok**, **OpenAI
GPT**). The nine per-thread outputs are then combined by majority vote in a
downstream aggregation step (see [Pipeline](#pipeline)).

> **Status:** anonymized for peer review. Author, institution, and funding details
> have been removed and are marked `[ANONYMIZED]` where a value is required.

---

## Repository structure

```
reddit-healthcareai-annotation/
├── README.md
├── requirements.txt
├── .env.example            # template for API keys (copy to .env)
├── .gitignore
├── LICENSE
├── src/
│   └── annotation/
│       ├── gemini/         # Gemini agent scripts
│       │   ├── annotate_gemini_hcp.py
│       │   ├── annotate_gemini_se.py
│       │   └── annotate_gemini_hr.py
│       ├── grok/           # Grok agent scripts
│       │   ├── annotate_grok_hcp.py
│       │   ├── annotate_grok_se.py
│       │   └── annotate_grok_hr.py
│       └── openai/         # OpenAI GPT agent scripts
│           ├── annotate_openai_hcp.py
│           ├── annotate_openai_se.py
│           └── annotate_openai_hr.py
├── data/                   # input threads go here (not distributed — see data/README.md)
│   └── README.md
├── results/                # annotation outputs are written here
└── docs/
    └── annotation_schema.md # input/output JSON schema and label definitions
```

### Script &rarr; agent &rarr; output map

| Model  | Agent | Script | Default output file |
|--------|-------|--------|---------------------|
| Gemini | HCP   | `src/annotation/gemini/annotate_gemini_hcp.py` | `results/annotated_reddit_10plus_FAST.json` |
| Gemini | SEng  | `src/annotation/gemini/annotate_gemini_se.py`  | `results/annotated_reddit_SOFTWARE_ENGINEERS.json` |
| Gemini | HCR   | `src/annotation/gemini/annotate_gemini_hr.py`  | `results/annotated_reddit_HEALTHCARE_RESEARCHERS.json` |
| Grok   | HCP   | `src/annotation/grok/annotate_grok_hcp.py`     | `results/annotated_reddit_GROK_HEALTHCARE_PROFESSIONALS_2.json` |
| Grok   | SEng  | `src/annotation/grok/annotate_grok_se.py`      | `results/annotated_reddit_GROK_SOFTWARE_ENGINEERS_2.json` |
| Grok   | HCR   | `src/annotation/grok/annotate_grok_hr.py`      | `results/annotated_reddit_GROK_HEALTHCARE_RESEARCHERS_2.json` |
| OpenAI | HCP   | `src/annotation/openai/annotate_openai_hcp.py` | `results/annotated_reddit_OPENAI_HEALTHCARE_PROFESSIONALS_2.json` |
| OpenAI | SEng  | `src/annotation/openai/annotate_openai_se.py`  | `results/annotated_reddit_OPENAI_SOFTWARE_ENGINEERS_2.json` |
| OpenAI | HCR   | `src/annotation/openai/annotate_openai_hr.py`  | `results/annotated_reddit_OPENAI_HEALTHCARE_RESEARCHERS_2.json` |

---

## Models

The exact model identifiers used in the reported runs are pinned inside each
script:

| Provider | Model identifier (in code)        | Sampling |
|----------|-----------------------------------|----------|
| Google   | `gemini-2.5-flash-lite`           | `temperature=0.2`, `top_p=0.95`, `top_k=40` |
| xAI      | `grok-4-1-fast-non-reasoning`     | `temperature=0.2` |
| OpenAI   | `gpt-4o-mini`                     | `temperature=0.2` |

To change a model, edit `MODEL_NAME` / `GROK_MODEL` / `OPENAI_MODEL` near the top
of the relevant script.

> **Note for the authors:** confirm these identifiers match the model names quoted
> in the manuscript before release, and reconcile any differences (e.g. `-flash`
> vs `-flash-lite`).

---

## Setup

Requires Python 3.9+.

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. provide API keys
cp .env.example .env      # then edit .env, or export the variables in your shell
```

The scripts read credentials from environment variables:

| Variable         | Used by            |
|------------------|--------------------|
| `GEMINI_API_KEY` | `annotate_gemini_*`|
| `XAI_API_KEY`    | `annotate_grok_*`  |
| `OPENAI_API_KEY` | `annotate_openai_*`|

If you use a `.env` file, load it before running (e.g. `export $(grep -v '^#' .env | xargs)`)
or install `python-dotenv` and load it at the top of each script.

---

## Input data

Place the collected Reddit threads at `data/reddit_threads.json` (or point the
`INPUT_FILE` environment variable at another path). The expected JSON structure
is:

```json
{
  "posts": [
    {"id": "abc123", "title": "...", "subreddit": "...", "selftext": "..."}
  ],
  "comments": {
    "abc123": [{"body": "...", "score": 42}]
  }
}
```

A thread is included for annotation only if it has at least one comment. Comments
are sorted by score (descending) and truncated to a 75,000-character budget
(~40 comments) per thread. See [`docs/annotation_schema.md`](docs/annotation_schema.md)
for full field definitions.

The dataset itself is **not** distributed in this repository — see
[`data/README.md`](data/README.md) for the reasons (Reddit content policy and
participant privacy) and for how to reconstruct it.

---

## Running an agent

Each script is standalone and interactive (it prints an estimate and asks for
confirmation before making API calls):

```bash
export INPUT_FILE=data/reddit_threads.json

python src/annotation/gemini/annotate_gemini_hcp.py
python src/annotation/grok/annotate_grok_se.py
python src/annotation/openai/annotate_openai_hr.py
# ... run all nine to produce the full set of outputs
```

Each run writes a JSON list to `results/`, one object per thread:

```json
{
  "post_id": "abc123",
  "subreddit": "medicine",
  "title": "...",
  "gemini_annotation": {
    "healthcare_professionals": true,
    "discusses_ai_in_healthcare": true,
    "discusses_ethical_implications": true,
    "ethical_dimensions": {
      "safety": true, "privacy": false, "bias": false,
      "transparency": true, "accountability": true
    },
    "themes": {"safety": ["over-reliance on AI triage"], "...": []}
  },
  "annotation_status": "success",
  "model_used": "gemini-2.5-flash-lite"
}
```

`annotation_status` is one of `success`, `json_error`, `rate_limit_error`, or
`error`. Failed calls are retried with exponential backoff before being recorded.

---

## Pipeline

```
data/reddit_threads.json
        │
        ▼
  9 annotation runs           (this repository)
  3 agents × 3 models
        │
        ▼
  results/*.json              (per-model, per-agent labels)
        │
        ▼
  majority-vote aggregation   ┐
  filtering to final corpus   │  downstream steps — add your
  distribution / co-occurrence│  aggregation & analysis scripts here
  human validation            ┘
```

The scripts in this repository cover the **annotation** stage only. The
majority-vote aggregation, corpus filtering, distributional/co-occurrence
analysis, and human-validation scripts are separate; add them under `src/` (for
example `src/aggregation/` and `src/analysis/`) so the package reproduces the
full study end to end.

---

## Reproducibility notes

- LLM outputs are not fully deterministic even at low temperature; exact counts
  may vary slightly between runs. The multi-model majority vote is designed to
  reduce sensitivity to any single model's variability.
- Provider model endpoints are periodically updated or retired. Pin dated model
  versions where your provider offers them if you need long-term reproducibility.
- Rate limits and pricing differ by provider and change over time; the per-thread
  delays in each script were tuned for the accounts used in the study.

---

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{[ANONYMIZED],
  title   = {Online Discussions to Requirements: Mining Stakeholder Perspectives to Support Healthcare AI Development},
  author  = {[ANONYMIZED]},
  journal = {[ANONYMIZED]},
  year    = {[ANONYMIZED]}
}
```

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). Update the copyright
holder after the anonymous review period.
