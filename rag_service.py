import os
import shutil
from typing import List, Optional
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
import time

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "research-rag-bge"
        
        # Check/Create Index
        # (Index creation logic remains same)
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768, 
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            except Exception as e:
                print(f"Index creation warning: {e}")

        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

        # Initialize LLM with correct parameters for Conversational Task
        self._llm_endpoint = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            task="text-generation", # Attempt text-generation first
            max_new_tokens=512,
            top_k=10,
            temperature=0.1,
            repetition_penalty=1.03,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        # Use ChatHuggingFace to handle chat models
        self.llm = ChatHuggingFace(llm=self._llm_endpoint)

        # Prompt for Strictly Grounded QA
        template = """You are an intelligent research assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Instructions:
1. Answer strictly based on the provided context.
2. If the question asks for a general summary or overview (e.g., "what is this paper about"), summarize the information available in the context (especially the Abstract or Introduction if present).
3. If the answer cannot be found in the context, reply: "This information is not available in the uploaded document."
4. Do not make up information.
"""
        self.prompt = PromptTemplate.from_template(template)

    async def process_pdf(self, file: UploadFile, session_id: str):
        # Save temp file
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Text Splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Add metadata for filtering by session/file
            for doc in splits:
                doc.metadata["session_id"] = session_id
                doc.metadata["source"] = file.filename

            # Add to Pinecone
            self.vectorstore.add_documents(documents=splits)
            
            return len(splits)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    async def process_url(self, url: str, session_id: str):
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
            
        for doc in splits:
            doc.metadata["session_id"] = session_id
            doc.metadata["source"] = url

        self.vectorstore.add_documents(documents=splits)
        return len(splits)

    def ask_question(self, question: str, session_id: str):
        try:
            # Retrieve context relevant to the session
            # increased k to 10 to capture more context for broad questions
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10, "filter": {"session_id": session_id}}
            )
            
            chain = (
                {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            
            return chain.invoke(question)
        except Exception as e:
            print(f"RAG Error: {e}")
            raise e

    def _format_docs(self, docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Singleton instance
rag_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service
