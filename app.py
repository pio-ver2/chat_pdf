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

# Estilo visual con temática oceánica
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;  /* Azul claro del océano */
            color: #004d40;  /* Texto en verde océano */
        }
        .stTitle {
            color: #0077b6;  /* Título en azul océano */
        }
        .stSubheader {
            color: #0077b6;  /* Subtítulos en azul oscuro */
        }
        .stButton>button {
            background-color: #00796b;  /* Botones de color verde océano */
            color: white;  /* Texto blanco en el botón */
        }
        .stImage>div>img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stSidebar {
            background-color: #b2dfdb;  /* Barra lateral de agua suave */
        }
        .stTextInput>div>div>input {
            background-color: #80d0c7;  /* Input de texto de fondo celeste */
        }
    </style>
""", unsafe_allow_html=True)

# Título de la aplicación con emojis
st.title("🌊 **Generación Aumentada por Recuperación (RAG)** 💬")

# Mostrar versión de Python
st.write("👨‍💻 **Versión de Python**:", platform.python_version())

# Cargar y mostrar imagen relacionada con el océano
try:
    image = Image.open('Chat_pdf.png')  # Asegúrate de que esta imagen esté disponible
    st.image(image, width=350)
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")

# Barra lateral
with st.sidebar:
    st.subheader("📝 **Este Agente te ayudará a realizar análisis sobre el PDF cargado**")
    st.write("""
    Sube un archivo PDF y pregunta sobre su contenido. El agente procesará el archivo y generará respuestas usando modelos de IA.
    """)

# Clave API de OpenAI
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("⚠️ Por favor ingresa tu clave de API de OpenAI para continuar")

# Cargar archivo PDF
pdf = st.file_uploader("📥 **Carga el archivo PDF**", type="pdf")

# Procesar el PDF
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"📝 **Texto extraído**: {len(text)} caracteres")
        
        # Dividir el texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"📄 **Documento dividido en** {len(chunks)} fragmentos")
        
        # Crear embeddings y base de conocimientos
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Interfaz para que el usuario ingrese su pregunta
        st.subheader("❓ **Escribe qué quieres saber sobre el documento**")
        user_question = st.text_area("🖋️ Escribe tu pregunta aquí...", placeholder="Escribe tu pregunta...")

        # Procesar pregunta cuando se envía
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Usar el modelo de OpenAI
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
🌊 **Acerca de la aplicación**: Esta aplicación utiliza modelos de IA para realizar preguntas y obtener respuestas basadas en documentos PDF cargados. 
Desarrollada con **Streamlit**, **OpenAI**, y **Langchain**. 🌟
""")
