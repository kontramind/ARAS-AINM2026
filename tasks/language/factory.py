import os
import sys
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

def get_llm() -> BaseChatModel:
    """
    Returns a ChatModel based on the LLM_PROVIDER env variable.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    # Get temperature from env, default to 0.0 for standard models, 
    # but strictly use 1.0 if using o1-preview/reasoning models.
    # We default to 1.0 here because your error explicitly requested it.
    temp = float(os.getenv("LLM_TEMPERATURE", "1.0"))
    
    print(f"🏭 Factory: Initializing LLM provider -> {provider.upper()} (Temp: {temp})")

    try:
        if provider == "azure":
            from langchain_openai import AzureChatOpenAI
            
            endpoint = os.getenv("AZURE_API_BASE")
            key = os.getenv("AZURE_API_KEY")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_MODEL")
            
            if not all([endpoint, key, deployment]):
                raise ValueError("❌ Azure config missing! Check AZURE_API_BASE, KEY, and DEPLOYMENT_MODEL.")

            return AzureChatOpenAI(
                azure_deployment=deployment,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                azure_endpoint=endpoint,
                api_key=key,
                temperature=temp 
            )

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=temp)

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=temp
            )
        
        else:
            raise ValueError(f"❌ Unsupported LLM_PROVIDER: {provider}")
            
    except ImportError as e:
        print(f"❌ Dependency Error: {e}")
        sys.exit(1)

def get_embeddings() -> Embeddings:
    """
    Returns an Embedding model based on configuration.
    Crucial: If LLM is Azure, we usually want Azure Embeddings too.
    """
    # Logic: If specific embedding provider set, use it. 
    # Otherwise, follow the LLM provider.
    provider = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")).lower()
    
    print(f"🏭 Factory: Initializing Embeddings -> {provider.upper()}")

    if provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if not deployment:
            raise ValueError("❌ Azure Embedding Deployment missing! Check AZURE_OPENAI_EMBEDDING_DEPLOYMENT.")

        return AzureOpenAIEmbeddings(
            azure_deployment=deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
        )

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()

    elif provider == "ollama" or provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # Runs 100% offline
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    else:
        raise ValueError(f"❌ Unsupported Embedding Provider: {provider}")