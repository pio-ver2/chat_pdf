import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform


st.markdown("""
    <style>
        body {
            background-color: #003366;  /* Azul marino oscuro */
            color: #ffffff;  /* Texto blanco para resaltar */
        }
        .stTitle {
            color: #80deea;  /* Azul océano claro para el título */
        }
        .stSubheader {
            color: #b2ebf2;  /* Azul más claro para los subtítulos */
        }
        .stButton>button {
            background-color: #004d40;  /* Verde océano oscuro para los botones */
            color: white;  /* Texto blanco en los botones */
        }
        .stImage>div>img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .stSidebar {
            background-color: #00897b;  /* Verde océano suave para el panel lateral */
        }
        .stTextInput>div>div>input {
            background-color: #4db6ac;  /* Fondo de los campos de texto en verde suave */
        }
        .stFileUploader>div>div>div>button {
            background-color: #00796b;  /* Botón de carga de archivo en verde oscuro */
            color: white;  /* Texto blanco */
        }
        .stMarkdown {
            color: #ffffff;  /* Texto de Markdown en blanco */
        }
    </style>
""", unsafe_allow_html=True)


st.title("🌊 **Generación Aumentada por Recuperación (RAG)** 💬")


st.write("👨‍💻 **Versión de Python**:", platform.python_version())


try:
    image = Image.open('Chat_pdf.png')  
    st.image(image, width=350)
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")


with st.sidebar:
    st.subheader("📝 **Este Agente te ayudará a realizar análisis sobre el PDF cargado**")
    st.write("""
    Sube un archivo PDF y pregunta sobre su contenido. El agente procesará el archivo y generará respuestas usando modelos de IA.
    """)


ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("⚠️ Por favor ingresa tu clave de API de OpenAI para continuar")


pdf = st.file_uploader("📥 **Carga el archivo PDF**", type="pdf")


if pdf is not None and ke:
    try:
        
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"📝 **Texto extraído**: {len(text)} caracteres")
        
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"📄 **Documento dividido en** {len(chunks)} fragmentos")
        
        
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        
        st.subheader("❓ **Escribe qué quieres saber sobre el documento**")
        user_question = st.text_area("🖋️ Escribe tu pregunta aquí...", placeholder="Escribe tu pregunta...")

        
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            
            llm = OpenAI(temperature=0, model_name="gpt-4")
            
            # Cargar el flujo de trabajo de pregunta y respuesta
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Ejecutar el flujo de trabajo
            response = chain.run(input_documents=docs, question=user_question)
            
            # Mostrar la respuesta
            st.markdown("### 📋 **Respuesta:**")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"❌ **Error al procesar el PDF**: {str(e)}")
        # Mostrar detalles del error para depuración
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ **Por favor ingresa tu clave de API de OpenAI para continuar**")
else:
    st.info("🔄 **Por favor carga un archivo PDF para comenzar**")

# Información adicional en el pie de página
st.markdown("---")
st.caption("""
🌊 **Acerca de la aplicación**: Esta aplicación utiliza **YOLOv5** para detección de objetos en imágenes capturadas con la cámara. 
Desarrollada con **Streamlit**, **OpenAI**, y **Langchain**. 🌟
""")
