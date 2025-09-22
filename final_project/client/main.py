import requests
import streamlit as st

st.set_page_config(page_title="DB & KB Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chatbot with Database & Knowledge Base Agents")

# -----------------------------
# API Endpoints
# -----------------------------
CHATBOT_API_URL = "http://localhost:8000/chatbot"
INGESTION_API_URL = "http://localhost:8000/ingestion-pipeline"

# -----------------------------
# Session State Initialization
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "upload_progress" not in st.session_state:
    st.session_state.upload_progress = 0

if "uploading_files" not in st.session_state:
    st.session_state.uploading_files = False

# -----------------------------
# Sidebar: File Upload
# -----------------------------
with st.sidebar:
    st.header("📂 Knowledge Base Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Files", type=["txt", "docx", "pdf"], accept_multiple_files=True
    )

    def ingest_files(files):
        st.session_state.uploading_files = True
        for i, f in enumerate(files, start=1):
            try:
                files_data = {"file": (f.name, f.read(), f.type)}
                response = requests.post(INGESTION_API_URL, files=files_data)
                if response.status_code == 200:
                    st.success(f"✅ {f.name} ingested successfully!")
                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"❌ Failed to ingest {f.name}: {detail}")
            except Exception as e:
                st.error(f"⚠️ Error ingesting {f.name}: {e}")
            st.session_state.upload_progress = i / len(files)
        st.balloons()
        st.session_state.upload_progress = 0
        st.session_state.uploading_files = False

    if uploaded_files:
        if not st.session_state.uploading_files:
            st.button("Start Upload", on_click=ingest_files, args=(uploaded_files,))
        if st.session_state.upload_progress > 0:
            st.progress(st.session_state.upload_progress)

# -----------------------------
# Chat Display
# -----------------------------
st.subheader("💬 Chat with Knowledge Base")

for idx, chat in enumerate(st.session_state.chat_history, start=1):
    st.chat_message("user").markdown(chat["question"])
    answer = chat["answer"].get("results", {})

    # Database agent
    if "sql_query" in answer:
        rows = answer.get("query_rows", [])
        if rows:
            headers = rows[0].keys()
            table = "| " + " | ".join(headers) + " |\n"
            table += "| " + " | ".join("---" for _ in headers) + " |\n"
            for row in rows:
                table += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
            st.chat_message("assistant").markdown(table)
        else:
            st.chat_message("assistant").markdown("No rows returned.")
        with st.expander(f"🔎 SQL Query (Q{idx})"):
            st.code(answer["sql_query"], language="sql")

    # Knowledge base agent
    elif "answer" in answer and "result" in answer["answer"]:
        st.chat_message("assistant").markdown(answer["answer"]["result"])

    else:
        st.chat_message("assistant").markdown("No response.")

# -----------------------------
# Chat Input / Controls at Bottom
# -----------------------------
st.divider()  # optional visual separator

with st.form("chat_form", clear_on_submit=True):
    q = st.text_input("Ask a question", key="chat_input")
    submitted = st.form_submit_button("Send")

if submitted and q.strip():
    with st.spinner("🤔 Thinking..."):
        try:
            response = requests.post(CHATBOT_API_URL, data={"query": q.strip()})
            response.raise_for_status()
            answer = response.json()
        except requests.exceptions.RequestException as e:
            answer = {"results": {"result": f"⚠️ Error: {e}"}}

        # Append to chat history
        st.session_state.chat_history.append({
            "question": q.strip(),
            "answer": answer
        })

    # Display the latest message immediately
    st.chat_message("user").markdown(q.strip())
    answer = st.session_state.chat_history[-1]["answer"].get("results", {})

    if "sql_query" in answer:
        rows = answer.get("query_rows", [])
        if rows:
            headers = rows[0].keys()
            table = "| " + " | ".join(headers) + " |\n"
            table += "| " + " | ".join("---" for _ in headers) + " |\n"
            for row in rows:
                table += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
            st.chat_message("assistant").markdown(table)
        else:
            st.chat_message("assistant").markdown("No rows returned.")
        with st.expander(f"🔎 SQL Query"):
            st.code(answer["sql_query"], language="sql")

    elif "answer" in answer and "result" in answer["answer"]:
        st.chat_message("assistant").markdown(answer["answer"]["result"])

# -----------------------------
# Clear Chat Button
# -----------------------------
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
