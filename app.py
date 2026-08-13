"""
Legal RAG System — Chainlit UI
Interactive frontend with real-time retrieval transparency and clickable citations.
"""

import os
import asyncio
from dotenv import load_dotenv
import chainlit as cl

os.environ["TRANSFORMERS_NO_TF"] = "1"

from legal_rag_system import (
    create_embedding_model,
    load_vector_store,
    create_retrieval_pipeline,
    build_rag_chain,
)


@cl.on_chat_start
async def on_chat_start():
    """Initialize the backend pipeline when a new chat session starts."""
    load_dotenv()
    
    if not os.environ.get("GOOGLE_API_KEY"):
        await cl.Message(content="⚠️ Error: GOOGLE_API_KEY is not set in the environment.").send()
        return

    loading_msg = cl.Message(content="⏳ **Initializing Legal RAG Engine...**\n\n> Loading embeddings, vector store, and case chunks. This may take a moment.")
    await loading_msg.send()
    
    def init_pipeline():
        embed_model = create_embedding_model()
        vstore = load_vector_store(embed_model)
        retriever = create_retrieval_pipeline(vstore)
        rag_chain = build_rag_chain(retriever)
        return retriever, rag_chain
    
    retriever, rag_chain = await asyncio.to_thread(init_pipeline)
    
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("rag_chain", rag_chain)
    
    welcome_text = """## ⚖️ Legal Assistant — Ready

**Hybrid Search** (BM25 + Semantic) · **Cross-Encoder Reranking** · **Gemini**

---

💬 **Try asking:**

- *Is a delay in email delivery a breach of contract?*
- *Are there cases about AI hiring discrimination and Title VII?*
- *What is the law on warrantless access to cloud-stored emails?*

> 📌 Every answer cites the specific cases it draws from — click on sources in the side panel to verify.
"""
    loading_msg.content = welcome_text
    await loading_msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user queries."""
    retriever = cl.user_session.get("retriever")
    rag_chain = cl.user_session.get("rag_chain")
    
    if not retriever or not rag_chain:
        await cl.Message(content="Pipeline not initialized correctly. Please refresh.").send()
        return
        
    query = message.content
    text_elements = []
    
    # Step 1: Search the case database
    async with cl.Step(name="🔍 Searching case database...", type="tool") as search_step:
        search_step.input = query
        search_step.output = "Running hybrid search (BM25 + Vector)..."
        
        retrieved_docs = await asyncio.to_thread(retriever.invoke, query)
        
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc.metadata
            case_name = meta.get("case_name", "Unknown Case")
            court = meta.get("court", "Unknown Court")
            date = meta.get("date", "Unknown Date")
            summary = meta.get("summary", "No summary available.")
            
            source_name = f"Source [{i}]: {case_name}"
            
            excerpt = doc.page_content[:600]
            if len(doc.page_content) > 600:
                excerpt += "..."
            
            content_body = (
                f"## 📋 Case Details\n\n"
                f"| Field | Info |\n"
                f"|-------|------|\n"
                f"| **🏛️ Court** | {court} |\n"
                f"| **📅 Date** | {date} |\n\n"
                f"**📝 Summary:** {summary}\n\n"
                f"---\n\n"
                f"### 📄 Relevant Excerpt\n\n"
                f"> {excerpt}\n"
            )
            
            text_el = cl.Text(name=source_name, content=content_body, display="side")
            text_elements.append(text_el)
            
        search_step.elements = text_elements
        search_step.output = f"✅ Found {len(retrieved_docs)} relevant precedent(s). Reranked by cross-encoder."

    # Step 2: Generate the answer
    async with cl.Step(name="⚖️ Analyzing precedents & drafting answer...", type="llm") as gen_step:
        gen_step.input = f"Generating grounded answer using {len(retrieved_docs)} retrieved case(s)..."
        
        msg = cl.Message(content="", elements=text_elements)
        
        async for chunk in rag_chain.astream(query):
            await msg.stream_token(chunk)
        
        # Append clickable source links at the bottom so the user can always reopen the side panel
        if text_elements:
            sources_footer = "\n\n---\n📚 **Sources** *(click to view)*\n\n"
            for el in text_elements:
                sources_footer += f"- {el.name}\n"
            msg.content += sources_footer
            
        await msg.send()
        gen_step.output = "✅ Answer generated with cited sources."
