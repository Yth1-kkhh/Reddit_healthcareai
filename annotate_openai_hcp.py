import os
import json
import time
from typing import List, Dict, Any

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bar: pip install tqdm")

# OpenAI SDK
try:
    from openai import OpenAI

    HAS_OPENAI = True
except Exception as e:
    HAS_OPENAI = False
    raise RuntimeError(
        "OpenAI SDK not found. Install with: pip install --upgrade openai"
    ) from e

# ============================================================================
# HEALTHCARE PROFESSIONALS AGENT - OPENAI
# ============================================================================
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-5-mini"

GENERATION_CONFIG = {
    'temperature': 0.2,
    'max_tokens': 1024,
}

# OpenAI client configuration
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Two-shot prompt for HEALTHCARE PROFESSIONALS
ANNOTATION_INSTRUCTION = """You are an expert annotator identifying healthcare professionals in Reddit discussions about AI ethics.

Analyze threads to determine:
1. Whether HEALTHCARE PROFESSIONALS are participating
2. Whether discussion involves AI/ML in healthcare
3. Which ethical dimensions are discussed with themes

HEALTHCARE PROFESSIONALS (true/false):
✓ TRUE: Physician, doctor, nurse, pharmacist, dentist, medical student (clinical), licensed clinician
✗ FALSE: Patients, family, general public, health IT without clinical role

DISCUSSES_AI_IN_HEALTHCARE (true/false):
✓ TRUE: AI/ML in medical/clinical/healthcare context
✗ FALSE: Generic AI or healthcare without AI

DISCUSSES_ETHICAL_IMPLICATIONS (true/false):
✓ TRUE: Ethics, risks, fairness, responsibility, trust concerns
✗ FALSE: Purely technical discussion

ETHICAL DIMENSIONS (true/false): Safety, Privacy, Bias, Transparency, Accountability
THEMES: 3-7 word phrases, specific not generic, empty [] if false

=== EXAMPLES ===

EXAMPLE 1:
Title: AI diagnostic tools - thoughts from the frontline?
Post: As an ER physician, I've been using AI-assisted triage. I'm worried about over-reliance. Last week system flagged patient as low-risk who had atypical MI.
Comment: ICU nurse here. Same concerns. AI doesn't see patient sweating.

OUTPUT:
{"healthcare_professionals":true,"discusses_ai_in_healthcare":true,"discusses_ethical_implications":true,"ethical_dimensions":{"safety":true,"privacy":false,"bias":false,"transparency":true,"accountability":true},"themes":{"safety":["over-reliance on AI triage","missed atypical presentations"],"privacy":[],"bias":[],"transparency":["black box algorithm concerns"],"accountability":["tool not replacement boundary"]}}

EXAMPLE 2:
Title: Should I trust AI diagnosis?
Post: I'm a patient who got AI diagnosis for skin condition. App said 85% melanoma. How accurate?
Comment: I'm concerned about privacy and who has access to medical images.

OUTPUT:
{"healthcare_professionals":false,"discusses_ai_in_healthcare":true,"discusses_ethical_implications":true,"ethical_dimensions":{"safety":true,"privacy":true,"bias":false,"transparency":false,"accountability":false},"themes":{"safety":["diagnostic accuracy doubts"],"privacy":["medical image data access"],"bias":[],"transparency":[],"accountability":[]}}

Valid JSON only, no markdown. Thread to analyze:
"""


def format_comments(comments: List[Dict], max_total_chars: int = 75000) -> str:
    """Format comments - 75K chars = ~40 comments"""
    if not comments:
        return "No comments."

    sorted_comments = sorted(comments, key=lambda x: x.get('score', 0), reverse=True)
    formatted = []
    total_chars = 0

    for i, comment in enumerate(sorted_comments):
        body = comment.get('body', 'N/A')
        if body in ['[deleted]', '[removed]', 'N/A']:
            continue

        comment_text = f"Comment {i + 1}:\n{body}"
        if total_chars + len(comment_text) > max_total_chars:
            break

        formatted.append(comment_text)
        total_chars += len(comment_text)

    return "\n\n".join(formatted)


def _strip_fences(s: str) -> str:
    """Strip ```json or ``` fences if the model wrapped the JSON."""
    s = s.strip()
    if s.startswith("```json"):
        s = s[len("```json"):].strip()
    if s.startswith("```"):
        s = s[len("```"):].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    return s


def _openai_chat_json(prompt: str, max_retries: int = 3) -> str:
    """Call OpenAI API with retry logic"""
    backoff = 4.0
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=GENERATION_CONFIG['temperature'],
                max_tokens=GENERATION_CONFIG['max_tokens'],
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            text = response.choices[0].message.content or ""
            return text

        except Exception as e:
            err_str = str(e)
            last_err = e

            transient = any(
                tok in err_str for tok in ["429", "Rate limit", "timeout", "temporarily unavailable", "overloaded"])
            if attempt < max_retries and transient:
                wait = backoff
                print(f"  ⏳ OpenAI error (attempt {attempt}/{max_retries}). Waiting {wait:.0f}s...")
                time.sleep(wait)
                backoff *= 1.8
                continue
            break

    raise last_err if last_err else RuntimeError("OpenAI call failed.")


def annotate_reddit_thread_openai(thread_data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Annotate a single Reddit thread using OpenAI API"""
    post_data = thread_data.get('post', thread_data)

    thread_text = f"""Title: {post_data.get('title', 'N/A')}
Subreddit: {post_data.get('subreddit', 'N/A')}
Post: {post_data.get('selftext', 'N/A')}

Comments:
{format_comments(thread_data.get('comments', []))}
"""

    full_prompt = f"{ANNOTATION_INSTRUCTION}\n{thread_text}"

    raw_text = ""

    for attempt in range(max_retries):
        try:
            raw_text = _openai_chat_json(full_prompt, max_retries=1)
            raw_text = _strip_fences(raw_text)

            annotation = json.loads(raw_text)

            annotated_thread = thread_data.copy()
            annotated_thread['openai_annotation'] = annotation
            annotated_thread['annotation_timestamp'] = time.time()
            annotated_thread['annotation_status'] = 'success'
            annotated_thread['model_used'] = OPENAI_MODEL
            annotated_thread['generation_config'] = GENERATION_CONFIG
            return annotated_thread

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            annotated_thread = thread_data.copy()
            annotated_thread['openai_annotation'] = {'error': 'JSON parsing failed'}
            annotated_thread['annotation_status'] = 'json_error'
            return annotated_thread

        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'Rate limit' in err_str:
                if attempt < max_retries - 1:
                    time.sleep(30)
                    continue
                annotated_thread = thread_data.copy()
                annotated_thread['openai_annotation'] = {'error': 'rate_limit'}
                annotated_thread['annotation_status'] = 'rate_limit_error'
                return annotated_thread
            else:
                annotated_thread = thread_data.copy()
                annotated_thread['openai_annotation'] = {'error': str(e)[:200]}
                annotated_thread['annotation_status'] = 'error'
                return annotated_thread

    return thread_data


def batch_annotate_threads(threads: List[Dict[str, Any]],
                           delay: float = 0.1) -> List[Dict[str, Any]]:
    """Annotate threads - NO CHECKPOINTS"""
    annotated_threads = []

    if HAS_TQDM:
        iterator = tqdm(threads, desc="Annotating (OpenAI-HCP)")
    else:
        iterator = threads

    for i, thread in enumerate(iterator):
        annotated = annotate_reddit_thread_openai(thread)
        annotated_threads.append(annotated)

        if not HAS_TQDM and i % 100 == 0:
            print(f"Processed {i + 1}/{len(threads)}...")

        if i < len(threads) - 1:
            time.sleep(delay)

    return annotated_threads


def save_annotated_threads(threads: List[Dict[str, Any]], output_file: str):
    """Save results"""
    output_data = []
    for thread in threads:
        post_data = thread.get('post', thread)
        output_thread = {
            'post_id': post_data.get('post_id') or post_data.get('id'),
            'subreddit': post_data.get('subreddit'),
            'title': post_data.get('title'),
            'openai_annotation': thread.get('openai_annotation'),
            'annotation_status': thread.get('annotation_status'),
            'model_used': thread.get('model_used')
        }
        output_data.append(output_thread)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved {len(output_data):,} threads to {output_file}")


def generate_summary_report(threads: List[Dict[str, Any]]):
    """Generate summary"""
    total = len(threads)
    successful = sum(1 for t in threads if t.get('annotation_status') == 'success')
    hcp_count = sum(1 for t in threads
                    if t.get('annotation_status') == 'success'
                    and t.get('openai_annotation', {}).get('healthcare_professionals', False))

    ai_hc_count = sum(1 for t in threads
                      if t.get('annotation_status') == 'success'
                      and t.get('openai_annotation', {}).get('discusses_ai_in_healthcare', False))

    ethics_count = sum(1 for t in threads
                       if t.get('annotation_status') == 'success'
                       and t.get('openai_annotation', {}).get('discusses_ethical_implications', False))

    print("\n" + "=" * 70)
    print("OPENAI ANNOTATION SUMMARY - HEALTHCARE PROFESSIONALS")
    print("=" * 70)
    print(f"Model: {OPENAI_MODEL}")
    print(f"Total: {total:,} | Success: {successful:,}")
    print(f"Healthcare Professionals: {hcp_count:,} ({hcp_count / successful * 100:.1f}%)")
    print(f"AI in Healthcare: {ai_hc_count:,} ({ai_hc_count / successful * 100:.1f}%)")
    print(f"Ethical Implications: {ethics_count:,} ({ethics_count / successful * 100:.1f}%)")
    print("=" * 70)


def load_threads_from_file(input_file: str) -> List[Dict[str, Any]]:
    """Load threads"""
    print(f"📂 Loading: {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    threads = []

    if isinstance(data, dict) and 'posts' in data and 'comments' in data:
        posts = data['posts']
        comments_dict = data['comments']

        for post in posts:
            post_id = post.get('id')
            comments = comments_dict.get(post_id, [])
            if comments:
                threads.append({'post': post, 'comments': comments})

        print(f"  ✓ Loaded {len(threads):,} threads")

    return threads


if __name__ == "__main__":
    input_file = os.environ.get('INPUT_FILE', 'data/reddit_threads.json')
    output_file = 'results/annotated_reddit_OPENAI_HEALTHCARE_PROFESSIONALS_2.json'

    print("\n" + "=" * 70)
    print("⚡ OPENAI - HEALTHCARE PROFESSIONALS ANNOTATION")
    print("=" * 70)
    print(f"Model: {OPENAI_MODEL}")
    print(f"Temperature: {GENERATION_CONFIG['temperature']}")
    print(f"Comment limit: 75K chars (~40 comments)")
    print(f"Delay: 0.1s")
    print("=" * 70 + "\n")

    threads = load_threads_from_file(input_file)

    print(f"\n📊 Processing {len(threads):,} threads")

    # Cost estimation for gpt-4o-mini
    avg_input_tokens = 3000
    avg_output_tokens = 300
    total_input_tokens = avg_input_tokens * len(threads)
    total_output_tokens = avg_output_tokens * len(threads)

    # gpt-4o-mini pricing: $0.150/1M input, $0.600/1M output
    input_cost = (total_input_tokens / 1_000_000) * 0.150
    output_cost = (total_output_tokens / 1_000_000) * 0.600
    total_cost = input_cost + output_cost

    avg_time_per_thread = 0.5
    total_time_hours = (len(threads) * avg_time_per_thread) / 3600

    print(f"⏱️  Estimated time: {total_time_hours:.1f} hours")
    print(f"💰 Estimated cost: ${total_cost:.2f}")
    print(f"    Input:  ${input_cost:.2f} ({total_input_tokens / 1_000_000:.1f}M tokens)")
    print(f"    Output: ${output_cost:.2f} ({total_output_tokens / 1_000_000:.1f}M tokens)\n")

    confirm = input(f"Process {len(threads):,} threads? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        exit()

    print(f"⏸️  Starting in 3 seconds...")
    time.sleep(3)

    try:
        annotated = batch_annotate_threads(threads, delay=0.1)
        save_annotated_threads(annotated, output_file)
        generate_summary_report(annotated)

        print(f"\n✅ COMPLETE! Saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        if annotated:
            partial_file = output_file.replace('.json', '_partial.json')
            save_annotated_threads(annotated, partial_file)
            print(f"💾 Saved partial results to: {partial_file}")
