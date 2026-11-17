import re
import time
from datetime import date
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from qdrant_client import QdrantClient

st.set_page_config(page_title="Rajen Vakil Transcripts AI", page_icon="🎙️", layout="centered")
st.title("🎙️ Ask Rajen Vakil's Talks")
st.caption("Search across Gujarati & English transcripts with exact timestamp citations.")

COLLECTION_NAME = "rajen_vakil_talks"
DAILY_GLOBAL_LIMIT = 200   # Max NEW uncached questions/day (~$0.40 CAD/day max on flash-lite)
SESSION_LIMIT = 20         # Max questions per browser session

FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]


@st.cache_resource
def get_clients():
    gemini = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    qdrant = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
    )
    return gemini, qdrant


@st.cache_resource
def get_daily_tracker():
    return {"date": date.today(), "count": 0}


gemini, qdrant = get_clients()
tracker = get_daily_tracker()

if tracker["date"] != date.today():
    tracker["date"] = date.today()
    tracker["count"] = 0


def normalize_query(q: str) -> str:
    # Collapse extra whitespace, lowercase, and strip trailing punctuation for higher cache hits
    cleaned = re.sub(r"\s+", " ", q.strip().lower())
    return cleaned.rstrip("?.!")


def generate_answer(prompt_text: str, sys_instruction: str) -> str:
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            resp = gemini.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    max_output_tokens=600,
                ),
            )
            return resp.text
        except APIError as e:
            last_error = e
            if e.code in (400, 404, 429, 500, 503, 504):
                time.sleep(0.5)
                continue
            raise
    raise last_error


@st.cache_data(ttl="7d", max_entries=1000, show_spinner=False)
def run_rag_cached(normalized_q: str, original_q: str) -> tuple[str, list[dict]]:
    """
    Caches the full RAG result (embedding + Qdrant hits + Gemini answer) across all users.
    If normalized_q was already asked within 7 days, returns immediately at $0.00 CAD cost.
    """
    # Count only actual API calls toward the daily spending cap
    if tracker["count"] >= DAILY_GLOBAL_LIMIT:
        raise RuntimeError("DAILY_LIMIT_REACHED")
    tracker["count"] += 1

    # 1. Embed user question
    q_emb = gemini.models.embed_content(
        model="gemini-embedding-001",
        contents=original_q,
        config={"output_dimensionality": 768},
    ).embeddings[0].values

    # 2. Retrieve top 4 matching segments from Qdrant
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=4,
        with_payload=True,
    ).points

    sources = []
    context_parts = []
    for idx, pt in enumerate(results, 1):
        p = pt.payload
        sources.append({
            "source": p["source"],
            "folder": p["folder"],
            "timestamp": p["timestamp"],
            "text": p["text"],
            "score": float(pt.score),
        })
        context_parts.append(
            f"[{idx}] Talk: {p['source']} (Topic/Folder: {p['folder']}) | Timestamp: [{p['timestamp']}]\n{p['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    # 3. Generate grounded response
    sys_instruction = (
        "You are a concise assistant answering questions strictly using the provided transcript excerpts "
        "from talks by Rajen Vakil (Gujarati and English).\n"
        "- Reply in the same language as the question.\n"
        "- Cite the Talk Title and [Timestamp] for every point.\n"
        "- If not found in the excerpts, say so briefly."
    )

    answer_text = generate_answer(
        prompt_text=f"Transcript Excerpts:\n\n{context_str}\n\nUser Question: {original_q}",
        sys_instruction=sys_instruction,
    )
    return answer_text, sources


if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask in English or Gujarati (e.g., What does he say about Patanjali Yogasutras?)"):
    if st.session_state.query_count >= SESSION_LIMIT:
        st.warning(f"Session limit reached ({SESSION_LIMIT} questions). Refresh the page to start a new session.")
        st.stop()

    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            norm_q = normalize_query(prompt)
            with st.spinner("Searching transcripts..."):
                answer_text, sources = run_rag_cached(norm_q, prompt)

            st.markdown(answer_text)

            with st.expander("📄 View Matched Transcript Excerpts"):
                for s in sources:
                    st.markdown(f"**{s['source']}** — `[{s['timestamp']}]` *(Relevance: {s['score']:.2f})*")
                    st.text(s["text"])
                    st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer_text})

        except RuntimeError as e:
            if "DAILY_LIMIT_REACHED" in str(e):
                st.warning("Daily new-question limit reached. Previously asked questions still work, or check back tomorrow!")
            else:
                st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"Temporary API error — please try again in a few seconds. ({e})")

