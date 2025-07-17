# app_rag.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- Configuración inicial ----------
st.set_page_config(page_title="Consulta tus Documentos con IA")
st.title("📄 Consulta tus documentos con IA local")

# Carga del modelo local
llm = Ollama(model="mistral")

plantilla = PromptTemplate.from_template(
    """Responde siempre en español, de forma clara y concisa.
    Usa solo la información proporcionada en el contexto si es posible.

    Contexto:
    {context}

    Pregunta:
    {question}
    """
)

# Crear carpeta para documentos cargados
if not os.path.exists("documentos"):
    os.makedirs("documentos")

# ---------- Subida del documento ----------
archivo = st.file_uploader("Sube un documento PDF", type="pdf")

if archivo:
    ruta_pdf = os.path.join("documentos", archivo.name)
    with open(ruta_pdf, "wb") as f:
        f.write(archivo.getbuffer())

    # Cargar y dividir el PDF
    loader = PyPDFLoader(ruta_pdf)
    paginas = loader.load()

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " "]
    )   
    documentos = splitter.split_documents(paginas)

    # Crear embeddings y base vectorial
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = FAISS.from_documents(documentos, embeddings)

    # Crear la cadena RAG
    cadena_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": plantilla}
    )

    # ---------- Interfaz de consulta ----------
    st.success("✅ Documento cargado correctamente. Ya puedes hacer preguntas.")
    pregunta = st.text_input("Escribe tu pregunta sobre el documento:")

    if pregunta:
        with st.spinner("Pensando..."):
            resultado = cadena_qa(pregunta)

        st.subheader("Respuesta")
        st.markdown(resultado["result"])

        st.subheader("Fragmentos utilizados")
        for i, doc in enumerate(resultado["source_documents"]):
            st.markdown(f"**Fragmento {i+1}:**\n{doc.page_content[:400]}...")
