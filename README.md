# Movie Analyzer

## Overview

The Movie Analyzer is a tool powered by LangChain that leverages advanced AI techniques to analyze, summarize, and provide insights into movie content. By integrating AI-driven models and data from movie databases, the Movie Analyzer enables users to interactively query and explore movie data, offering deep insights into plot structure, character development, thematic elements, and much more.

## Features

- Movie Summary: Get a brief summary of any movie by providing its title or IMDb ID.
- Sentiment Analysis: Analyze the sentiment of movie reviews and summaries.
- Genre & Theme Extraction: Extract and identify genres and key themes from movie descriptions.
- Character Analysis: Identify key characters, their traits, and relationships within the movie.
- Plot Breakdown: Provide a detailed breakdown of the plot with insights into major plot points and twists.
- Custom Queries: Ask custom questions about a movie (e.g., “Who is the villain?” or “What are the key plot twists?”).

## Tech Stack
- Frontend - Streamlit
- Vector Database - Chroma DB
- Backend - Python
- Agents - Langchain with MistralAI

## Installation
- Clone the Repo
    ```bash
    git clone https://github.com/HeathKnowles/MovieAnalyzer
    ```
- Create a virtual environment
    ```python
    python - m venv env
- Install the requirements
    ```bash
    pip install -r requirements.txt
    ```
- Add the .env folder and keep the API Keys as in **.env.example**

- Run the streamlit APP
    ```bash
    streamlit run app.py
    ```

- Type the Movie Name in the app window

- Ask away anything about the movie