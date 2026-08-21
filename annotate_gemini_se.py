try:
    import google.generativeai as genai

    USE_NEW_SDK = False
except ImportError:
    from google import genai

    USE_NEW_SDK = True

import json
import time
from typing import List, Dict, Any
import os

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================================
# SOFTWARE ENGINEERS AGENT - OPTIMIZED FOR SPEED
# ============================================================================
MODEL_NAME = 'gemini-2.5-flash-lite'

GENERATION_CONFIG = {
    'temperature': 0.2,
    'top_p': 0.95,
    'top_k': 40,
    'max_output_tokens': 1024,
}

# Configure Gemini API
if USE_NEW_SDK:
    client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
else:
    genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

# Two-shot prompt for SOFTWARE ENGINEERS
GEMINI_INSTRUCTION = """You are an expert annotator identifying software engineers in Reddit discussions about AI ethics.

Analyze threads to determine:
1. Whether SOFTWARE ENGINEERS are participating
2. Whether discussion involves AI/ML in healthcare
3. Which ethical dimensions are discussed with themes

SOFTWARE ENGINEERS (true/false):
✓ TRUE: ML engineer, software developer, data scientist, AI researcher WITH active technical involvement
  - Must show: "As an ML engineer...", "I'm a developer...", "When we deployed...", "Our training pipeline..."
  - Active involvement in development/training/deployment/implementation
✗ FALSE: Passive tech users, generic tech mentions, "developers should...", non-technical roles

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
Title: Deploying medical AI - lessons learned
Post: As an ML engineer, we just deployed a diagnostic model in production. Major issue: our training data had significant demographic imbalance. The model performs worse for minority patients. We're scrambling to fix this before regulatory review.
Comment: Senior dev here. Same problem. We now require fairness audits before any healthcare deployment.

OUTPUT:
{"software_engineers":true,"discusses_ai_in_healthcare":true,"discusses_ethical_implications":true,"ethical_dimensions":{"safety":false,"privacy":false,"bias":true,"transparency":false,"accountability":true},"themes":{"safety":[],"privacy":[],"bias":["training data demographic imbalance","worse performance for minorities"],"transparency":[],"accountability":["regulatory review requirements","mandatory fairness audits"]}}

EXAMPLE 2:
Title: AI medical apps - what do you think?
Post: I'm a patient using an AI health app. It gave me a diagnosis but I don't understand how it works. Should I trust it?
Comment: I'm also concerned about where my health data goes. Who has access?

OUTPUT:
{"software_engineers":false,"discusses_ai_in_healthcare":true,"discusses_ethical_implications":true,"ethical_dimensions":{"safety":true,"privacy":true,"bias":false,"transparency":true,"accountability":false},"themes":{"safety":["trusting AI diagnosis"],"privacy":["health data access concerns"],"bias":[],"transparency":["unexplainable AI decisions"],"accountability":[]}}

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


def annotate_reddit_thread(thread_data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
    """Annotate single thread"""
    post_data = thread_data.get('post', thread_data)

    thread_text = f"""Title: {post_data.get('title', 'N/A')}
Subreddit: {post_data.get('subreddit', 'N/A')}
Post: {post_data.get('selftext', 'N/A')}

Comments:
{format_comments(thread_data.get('comments', []))}
"""

    full_prompt = f"{GEMINI_INSTRUCTION}\n{thread_text}"

    for attempt in range(max_retries):
        try:
            if USE_NEW_SDK:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=full_prompt,
                    config=GENERATION_CONFIG
                )
                annotation_text = response.text
            else:
                model = genai.GenerativeModel(MODEL_NAME, generation_config=GENERATION_CONFIG)
                response = model.generate_content(full_prompt)
                annotation_text = response.text

            # Clean JSON
            annotation_text = annotation_text.strip()
            if annotation_text.startswith('```json'):
                annotation_text = annotation_text[7:]
            if annotation_text.startswith('```'):
                annotation_text = annotation_text[3:]
            if annotation_text.endswith('```'):
                annotation_text = annotation_text[:-3]
            annotation_text = annotation_text.strip()

            annotation = json.loads(annotation_text)

            annotated_thread = thread_data.copy()
            annotated_thread['gemini_annotation'] = annotation
            annotated_thread['annotation_timestamp'] = time.time()
            annotated_thread['annotation_status'] = 'success'
            annotated_thread['model_used'] = MODEL_NAME

            return annotated_thread

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            annotated_thread = thread_data.copy()
            annotated_thread['gemini_annotation'] = {'error': 'JSON parsing failed'}
            annotated_thread['annotation_status'] = 'json_error'
            return annotated_thread

        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                if attempt < max_retries - 1:
                    time.sleep(30)
                    continue
                annotated_thread = thread_data.copy()
                annotated_thread['gemini_annotation'] = {'error': 'rate_limit'}
                annotated_thread['annotation_status'] = 'rate_limit_error'
                return annotated_thread
            else:
                annotated_thread = thread_data.copy()
                annotated_thread['gemini_annotation'] = {'error': str(e)[:200]}
                annotated_thread['annotation_status'] = 'error'
                return annotated_thread

    return thread_data


def batch_annotate_threads(threads: List[Dict[str, Any]],
                           delay: float = 0.05) -> List[Dict[str, Any]]:
    """Annotate threads with minimal delay"""
    annotated_threads = []

    if HAS_TQDM:
        iterator = tqdm(threads, desc="Annotating (SWE)")
    else:
        iterator = threads

    for i, thread in enumerate(iterator):
        annotated = annotate_reddit_thread(thread)
        annotated_threads.append(annotated)

        if not HAS_TQDM and i % 100 == 0:
            print(f"Processed {i}/{len(threads)}...")

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
            'gemini_annotation': thread.get('gemini_annotation'),
            'annotation_status': thread.get('annotation_status'),
            'model_used': thread.get('model_used')
        }
        output_data.append(output_thread)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved {len(output_data)} threads to {output_file}")


def generate_summary_report(threads: List[Dict[str, Any]]):
    """Generate summary"""
    total = len(threads)
    successful = sum(1 for t in threads if t.get('annotation_status') == 'success')
    swe_count = sum(1 for t in threads
                    if t.get('annotation_status') == 'success'
                    and t.get('gemini_annotation', {}).get('software_engineers', False))

    ai_hc_count = sum(1 for t in threads
                      if t.get('annotation_status') == 'success'
                      and t.get('gemini_annotation', {}).get('discusses_ai_in_healthcare', False))

    ethics_count = sum(1 for t in threads
                       if t.get('annotation_status') == 'success'
                       and t.get('gemini_annotation', {}).get('discusses_ethical_implications', False))

    print("\n" + "=" * 70)
    print("SOFTWARE ENGINEERS ANNOTATION SUMMARY")
    print("=" * 70)
    print(f"Total: {total} | Success: {successful}")
    print(f"Software Engineers: {swe_count} ({swe_count / successful * 100:.1f}%)")
    print(f"AI in Healthcare: {ai_hc_count} ({ai_hc_count / successful * 100:.1f}%)")
    print(f"Ethical Implications: {ethics_count} ({ethics_count / successful * 100:.1f}%)")
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

        print(f"  ✓ Loaded {len(threads)} threads")

    return threads


if __name__ == "__main__":
    input_file = os.environ.get('INPUT_FILE', 'data/reddit_threads.json')
    output_file = 'results/annotated_reddit_SOFTWARE_ENGINEERS.json'

    print("\n" + "=" * 70)
    print("⚡ SOFTWARE ENGINEERS ANNOTATION")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Comment limit: 75K chars (~40 comments)")
    print(f"Delay: 0.05s")
    print("=" * 70 + "\n")

    threads = load_threads_from_file(input_file)

    print(f"\n📊 Processing {len(threads)} threads")

    avg_time_per_thread = 0.8
    total_time_hours = (len(threads) * avg_time_per_thread) / 3600

    print(f"⏱️  Estimated time: {total_time_hours:.1f} hours")
    print(f"💰 Estimated cost: ~${len(threads) * 0.00025:.2f}\n")

    confirm = input(f"Process {len(threads)} threads? (y/n): ")
    if confirm.lower() != 'y':
        print("Cancelled.")
        exit()

    print(f"⏸️  Starting in 3 seconds...")
    time.sleep(3)

    try:
        annotated = batch_annotate_threads(threads, delay=0.05)
        save_annotated_threads(annotated, output_file)
        generate_summary_report(annotated)
        print(f"\n✅ COMPLETE! Saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        if annotated:
            save_annotated_threads(annotated, output_file.replace('.json', '_partial.json'))
