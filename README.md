# 🧠 Consulta tus documentos PDF con IA local (RAG + Streamlit)

Este proyecto permite subir un archivo PDF y hacerle preguntas en lenguaje natural, utilizando un sistema **RAG (Retrieval-Augmented Generation)** totalmente local. El modelo de lenguaje se ejecuta mediante **Ollama**, usando **LangChain** como framework y **Streamlit** como interfaz web.

📽️ **Este código fue presentado en un taller práctico** sobre sistemas RAG en local, en colaboración con **María Jesús Puerta Angulo** y el Club de los Frikis 🔥.

🔗 Mira el taller completo aquí:
👉 [https://www.youtube.com/watch?v=i8m2kKP0bSQ](https://www.youtube.com/watch?v=i8m2kKP0bSQ)

---

## 🚀 ¿Qué hace esta app?

* Permite **subir un documento PDF**.
* Divide el contenido en **chunks solapados** (fragmentos).
* Genera **embeddings** usando un modelo local (Mistral vía Ollama).
* Almacena los vectores en **FAISS** para recuperación eficiente.
* Usa un modelo LLM local para **responder preguntas** basándose en el contenido real del documento.
* Todo con una interfaz amigable vía **Streamlit**.

---

## ⚙️ Requisitos

* Python 3.10+
* Ollama instalado y corriendo el modelo `mistral`
* Dependencias de Python:

```bash
pip install streamlit langchain faiss-cpu
```

> Nota: Se recomienda instalar también `langchain-community` si no está incluido en tu entorno.

---

## 🧩 Estructura del código

### Carga y preprocesado del PDF

```python
loader = PyPDFLoader(ruta_pdf)
paginas = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " "]
)
documentos = splitter.split_documents(paginas)
```

Se divide el texto en fragmentos superpuestos, lo que ayuda a que el modelo tenga suficiente contexto sin perder coherencia.

---

### Embeddings y base vectorial

```python
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = FAISS.from_documents(documentos, embeddings)
```

Se genera una base vectorial para poder recuperar los fragmentos más relevantes a cada pregunta.

---

### Consulta y respuesta

```python
cadena_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": plantilla}
)
```

El sistema responde con precisión usando solo la información del documento subido, gracias al modelo local y a la plantilla personalizada.

---

## 🖥️ Ejecutar la app

1. Inicia tu servidor Ollama con el modelo `mistral` cargado:

```bash
ollama run mistral
```

2. Lanza la app:

```bash
streamlit run app_rag.py
```

3. Accede en tu navegador a `http://localhost:8501`, sube tu PDF y haz preguntas.

---

## 🧪 Demo del Taller

🔍 Este proyecto fue parte del taller **"Cómo instalar un sistema RAG en local paso a paso"**, presentado junto al Club de los Frikis.

🎬 **[Ver el video del taller en YouTube](https://www.youtube.com/watch?v=i8m2kKP0bSQ)**

---

## 📩 Contacto

¿Dudas o quieres adaptar este sistema a tus propios documentos o dominio?
Contáctame en [LinkedIn](https://www.linkedin.com/in/matias-palomino-luna24/) o deja un comentario en el video.
