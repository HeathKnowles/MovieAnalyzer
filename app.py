import os
from dotenv import load_dotenv
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import wikipediaapi
import chromadb
from chromadb.config import Settings


load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRALAI_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY is not set. Please check your .env file.")
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

def setup_mistral_model():
    return ChatMistralAI(model="mistral-large-latest")

model = setup_mistral_model()

chroma_client = None

def initialize_chroma_db():
    global chroma_client
    
    if chroma_client is None:
        settings = Settings(
            persist_directory=".chroma_data",  # Ensure persistence directory is consistent
            anonymized_telemetry=False        # Disable telemetry for simplicity
        )
        chroma_client = chromadb.Client(settings)
    
    if "movie_plots" in [collection.name for collection in chroma_client.list_collections()]:
        collection = chroma_client.get_collection("movie_plots")
    else:
        collection = chroma_client.create_collection("movie_plots")
    
    return collection


chroma_collection = initialize_chroma_db()

def create_chain():
    template = """Answer the following question based on the provided context:
Context: {context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    setup_and_input = RunnableParallel(
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    )

    return setup_and_input | prompt | model | output_parser

chain = create_chain()

st.title("🎬 Movie Analyzer with Mistral AI and Chroma DB")
st.markdown("Retrieve movie data from Wikipedia and ask questions interactively!")

if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.movie_title = None
    st.session_state.plot_content = None

movie_title = st.text_input("Enter a movie title:", placeholder="E.g., The Matrix", key="movie_title_input")

if st.button("Get Plot and Start Chat"):
    if movie_title:
        wiki_wiki = wikipediaapi.Wikipedia('MovieAnalyzer (yyashuday13@gmail.com)','en')
        page = wiki_wiki.page(movie_title)
        if page.exists():
            plot_section = page.section_by_title("Plot")
            if plot_section:
                plot_text = plot_section.text
                unique_id = movie_title.lower().replace(" ", "_")  # Generate a unique ID
                
                chroma_collection.add(
                    ids=[unique_id],
                    documents=[plot_text],
                    metadatas=[{"title": movie_title}]
                )
                
                st.session_state.movie_title = movie_title
                st.session_state.plot_content = plot_text
                st.session_state.history = [("Bot", f"Let's chat about the movie: {movie_title}. Ask me anything about its plot or details.")]
                st.success(f"Plot for '{movie_title}' fetched and stored in Chroma DB successfully!")
            else:
                st.error("No 'Plot' section found.")
        else:
            st.error(f"No Wikipedia page found for '{movie_title}'.")

if st.session_state.movie_title:
    st.write(f"### Chat with the Movie: {st.session_state.movie_title}")
    
    user_question = st.text_input("Ask a question about the plot:", key="user_question_input")

    if st.button("Ask Question"):
        if user_question:
            results = chroma_collection.query(
                query_texts=[user_question],
                n_results=1
            )
            context = results['documents'][0] if results['documents'] else "Sorry, I couldn't find relevant information."
            
            input_data = {
                "context": context,
                "question": user_question
            }
            answer = chain.invoke(input_data)
            
            st.session_state.history.append(("User", user_question))
            st.session_state.history.append(("Bot", answer))
            
            for sender, message in st.session_state.history:
                st.write(f"**{sender}:** {message}")
    
    st.markdown("---")
    st.write("Powered by LangChain, Mistral AI, Chroma DB, and Streamlit.")
