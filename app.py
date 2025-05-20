import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import streamlit.components.v1 as components


# --- Load environment ---
load_dotenv()
st.set_page_config(page_title="FirstPage AI", layout="wide")

# --- Load RAG system ---
@st.cache_resource
def load_knowledge_chain():
    loader = TextLoader("LandingPageGuide.md", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever()
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.5)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = load_knowledge_chain()

# --- Prompt Builder ---
def build_prompt(name, industry, goal, audience, problem, usp, offer):
    return f"""
You are a world-class landing page copywriter and strategist.

Your job is to write a complete, high-converting landing page COPY based on the user inputs — not just structure. This copy should be ready for someone to paste directly into a page builder like Carrd, Webflow, or Framer.

Use best practices from conversion copywriting, landing page psychology, and proven design patterns. Each section should be written out as actual content, not a description of what to include.

Inputs:
- Name: {name}
- Industry: {industry}
- Goal: {goal}
- Target Audience: {audience}
- Problem Solved: {problem}
- Unique Value Proposition: {usp or 'N/A'}
- Offer/Incentive: {offer or 'N/A'}

Format your response like this:

---

### 1. COPY

**Headline:**  
[Actual headline text]

**Subheadline:**  
[Actual subheadline]

**Primary CTA Button Text:**  
[CTA copy]

**Section Copy:**  
Write 4–6 sections of actual landing page content, ready to paste:
- Section 1 Title:  
[Full copy here]

- Section 2 Title:  
[Full copy here]

...continue as needed...

---

### 2. DESIGN DIRECTION

**Layout Style:**  
[Best layout suggestion]

**Color Palette & Imagery:**  
[Recommended style]

**Mobile Tips:**  
[UX ideas]

---

### 3. STRATEGIC GUIDANCE

**Top Mistakes to Avoid:**  
[Bullet points]

**Build Order:**  
[Steps: write → wireframe → design → build]

**Recommended Tools:**  
[3 tools suited to the goal and user type]
"""

# --- UI ---
st.title("🔮 FirstPage AI")
st.caption("Turn your idea into a complete landing page — copy, structure, and expert tips included.")

left, right = st.columns([1, 2])

with left:
    st.header("💡 Input Details")
    name = st.text_input("Product or Service Name")
    industry = st.text_input("Industry / Niche")
    goal = st.text_input("Primary Goal (e.g. signups, purchases)")
    audience = st.text_input("Target Audience")
    problem = st.text_area("Problem it Solves")
    usp = st.text_input("Unique Value Proposition (optional)")
    offer = st.text_input("Offer / Incentive (optional)")

    submitted = st.button("Generate Landing Page")

# --- Run Prompt & Show Output ---
with right:
    if submitted:
        if not all([name, industry, goal, audience, problem]):
            st.warning("Please fill out all required fields (Name, Industry, Goal, Audience, Problem).")
        else:
            with st.spinner("Crafting your landing page copy..."):
                prompt = build_prompt(name, industry, goal, audience, problem, usp, offer)
                result = qa_chain.run(prompt)

            components.html("""
                <script>
                    // Add a longer delay to ensure the content is fully rendered and the DOM is updated
                    setTimeout(function() {
                        // Scroll the entire window to the top
                        window.parent.scrollTo({ top: 0, behavior: 'smooth' });
                    }, 500); // Increased delay to 500 milliseconds (adjust if needed)
                </script>
            """, height=0)

            
            st.subheader("✅ Your Landing Page Copy & Blueprint")
            st.markdown(result)
