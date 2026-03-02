from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import our agnostic factory
from tasks.language.factory import get_llm, get_embeddings

class RAGPipeline:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.llm = get_llm()
        self.embeddings = get_embeddings()
        self.persist_directory = persist_directory
        self.vectorstore = None

        print(f"✅ RAG Initialized. LLM: {self.llm.__class__.__name__} | Embeddings: {self.embeddings.__class__.__name__}")

    def ingest_texts(self, texts: list[str], metadatas: list[dict] = None):
        """Creates an offline vector database from raw texts."""
        print("⚙️ Embedding documents...")
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("✅ Ingestion complete.")

    def build_chain(self):
        """Builds the actual retrieval and generation chain."""
        if not self.vectorstore:
            # If no DB is loaded in memory, try to load it from disk
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Standard RAG Prompt (can be customized for Medical, Code, etc.)
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        prompt = PromptTemplate.from_template(template)

        # Formatting helper
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Build the chain using LangChain Expression Language (LCEL)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str) -> str:
        """Runs a question through the RAG chain."""
        chain = self.build_chain()
        return chain.invoke(question)

# --- Quick Test ---
if __name__ == "__main__":
    rag = RAGPipeline()
    # Dummy data representing an Emergency Room guideline
    rag.ingest_texts([
        "Patient exhibits severe chest pain and shortness of breath. Immediate triage level 1 required.",
        "A sprained ankle should be iced and elevated. Triage level 4."
    ])
    
    response = rag.query("What should I do for a patient with severe chest pain?")
    print(f"\n🤖 Response: {response}")