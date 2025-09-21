import streamlit as st

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Ingestion", "Chatbot", "Documents"])

# --- Main app content ---
st.title("📌 Document Ingestion Chatbot")

if page == "Ingestion":
    st.header("📥 Data Ingestion")
    st.write("Upload and preprocess your data here.")
    
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "pdf"])
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        # Add ingestion logic here (parse, store, etc.)

elif page == "Chatbot":
    st.header("🤖 Chatbot")
    st.write("Ask me anything!")

    user_input = st.text_input("Your message:")
    if st.button("Send"):
        if user_input.strip():
            st.write(f"**You:** {user_input}")
            # Dummy response (replace with real LLM logic)
            st.write("**Bot:** This is a sample response.")
        else:
            st.warning("Please enter a message.")

elif page == "Documents":
    st.header("📂 Documents")
    st.write("Manage and explore your documents here.")

    st.text_area("Paste text or notes here")
    st.button("Save Document")
