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
DAILY_GLOBAL_LIMIT = 500   # Max total questions/day across all visitors (~$0.35 CAD/day max)
SESSION_LIMIT = 25         # Max questions per individual browser session

FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
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

# Reset global daily counter when a new UTC day starts
if tracker["date"] != date.today():
    tracker["date"] = date.today()
    tracker["count"] = 0


def stream_with_fallback(prompt_text: str, sys_instruction: str):
    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            stream = gemini.models.generate_content_stream(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
            return
        except APIError as e:
            last_error = e
            if e.code in (400, 404, 429, 500, 503, 504):
                time.sleep(0.5)
                continue
            raise
    raise last_error


if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask in English or Gujarati (e.g., What does he say about Patanjali Yogasutras?)"):
    if tracker["count"] >= DAILY_GLOBAL_LIMIT:
        st.warning("Daily community question limit reached. Please check back tomorrow!")
        st.stop()

    if st.session_state.query_count >= SESSION_LIMIT:
        st.warning(f"Session limit reached ({SESSION_LIMIT} questions). Refresh the page to start a new session.")
        st.stop()

    tracker["count"] += 1
    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching transcripts..."):
                # 1. Embed user question
                q_emb = gemini.models.embed_content(
                    model="gemini-embedding-001",
                    contents=prompt,
                    config={"output_dimensionality": 768},
                ).embeddings[0].values

                # 2. Retrieve top 6 matching segments from Qdrant
                results = qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=q_emb,
                    limit=6,
                    with_payload=True,
                ).points

                context_parts = []
                for idx, pt in enumerate(results, 1):
                    p = pt.payload
                    context_parts.append(
                        f"[{idx}] Talk: {p['source']} (Topic/Folder: {p['folder']}) | Timestamp: [{p['timestamp']}]\n{p['text']}"
                    )
                context_str = "\n\n---\n\n".join(context_parts)

            sys_instruction = (
                "You are a helpful assistant answering questions strictly using the provided transcript excerpts "
                "from talks by Rajen Vakil. The transcripts contain Gujarati and English.\n"
                "- Reply in the same language the user asked their question in, unless they ask otherwise.\n"
                "- Cite the exact Talk Title and [Timestamp] for every point you make.\n"
                "- If the excerpts do not contain the answer, clearly state that it wasn't found in the transcripts."
            )

            full_text = st.write_stream(
                stream_with_fallback(
                    prompt_text=f"Transcript Excerpts:\n\n{context_str}\n\nUser Question: {prompt}",
                    sys_instruction=sys_instruction,
                )
            )

            with st.expander("📄 View Matched Transcript Excerpts"):
                for pt in results:
                    p = pt.payload
                    st.markdown(f"**{p['source']}** — `[{p['timestamp']}]` *(Relevance: {pt.score:.2f})*")
                    st.text(p["text"])
                    st.divider()

            st.session_state.messages.append({"role": "assistant", "content": full_text})

        except Exception as e:
            st.error(f"Temporary API error — please try again in a few seconds. ({e})")

