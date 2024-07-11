# botmovie

# Movie Recommendation Chatbot

## Overview
This project is a Movie Recommendation Chatbot developed using FastAPI as the backend framework and Streamlit as the frontend interface. It leverages the power of OpenAI's GPT-3 model and HuggingFace's sentence-transformers for embeddings, integrated with a ChromaDB vector database for efficient storage and retrieval of movie data.


## Frameworks and Languages
- **Backend Framework**: FastAPI, OpenAI, ChromaDB, TMDB API Databse
- **Frontend Interface**: Streamlit
- **Languages**: Python

## Basic Usage
The Movie Recommendation Chatbot allows users to input a query about movies and get personalized movie recommendations based on the input. The backend processes the query using a language model and retrieves relevant movie information from a vector database.

## Project Description
This project uses FastAPI for handling API requests, and Streamlit for providing a simple user interface. The core components include:
- **OpenAI's GPT-3** for processing user queries.
- **HuggingFace's sentence-transformers** for creating embeddings of movie plots.
- **ChromaDB** for storing and querying movie data using embeddings.

## Code Explanation

### FastAPI Backend (`main.py`)
1. **Imports and Initialization**:
    ```python
    import os
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from langchain_openai import OpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from chromadb import Client
    from chromadb.config import Settings
    import logging
    from dotenv import load_dotenv

    app = FastAPI()
    load_dotenv()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ```

    - Import necessary libraries and configure logging.
    - Initialize FastAPI app and load environment variables.

2. **Initialize Models and Database**:
    ```python
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("Please set the OPENAI_API_KEY environment variable.")
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    try:
        llm = OpenAI(api_key=openai_api_key, model_name="text-davinci-003")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Successfully initialized LLM and embeddings.")
    except Exception as e:
        logger.error(f"Error initializing LLM or embeddings: {e}")
        raise

    try:
        client = Client(Settings(persist_directory="chroma_db"))
        collection = client.create_collection("movies")
        logger.info("Successfully initialized ChromaDB.")
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        raise
    ```

    - Retrieve OpenAI API key from environment variables and initialize the language model and embeddings.
    - Initialize ChromaDB client and create a collection for storing movie data.

3. **Add Movie Data to ChromaDB**:
    ```python
    movie_data = [
        {"title": "Inception", "genre": "Sci-Fi", "plot": "A thief who steals corporate secrets through the use of dream-sharing technology."},
        {"title": "The Matrix", "genre": "Action", "plot": "A computer hacker learns about the true nature of reality and his role in the war against its controllers."},
        {"title": "Jurassic Park", "genre": "Sci-Fi", "plot": "A theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok."},
        {"title": "The Godfather", "genre": "Crime", "plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."},
        {"title": "Pulp Fiction", "genre": "Crime", "plot": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
    ]

    try:
        for idx, movie in enumerate(movie_data):
            text_to_embed = f"{movie['plot']} [GENRE: {movie['genre']}]"
            plot_embedding = embeddings.embed_documents([text_to_embed])[0]
            collection.add(ids=[str(idx)], embeddings=[plot_embedding], metadatas=[movie], documents=[text_to_embed])
        logger.info("Successfully added movie data to ChromaDB.")
    except Exception as e:
        logger.error(f"Error adding movie data to ChromaDB: {e}")
        raise
    ```

    - Define example movie data.
    - Embed movie plots and add them to ChromaDB.

4. **API Endpoint for Movie Recommendations**:
    ```python
    class UserQuery(BaseModel):
        query: str = None

    @app.post("/movies/")
    async def get_movies(user_query: UserQuery):
        try:
            if user_query.query:
                logger.info(f"Received query: {user_query.query}")
                query_embedding = embeddings.embed_documents([user_query.query])[0]
                results = collection.query(query_embeddings=[query_embedding], n_results=3)
                recommendations = []
                for metadata_list in results.get('metadatas', []):
                    recommendations.extend(metadata_list)

                if not recommendations:
                    logger.info("No recommendations found.")
                    raise HTTPException(status_code=404, detail="No recommendations found")

                return {"movies": recommendations[:2]}
            else:
                results = collection.get()
                all_movies = results.get("metadatas", [])
                return {"movies": all_movies}
        except Exception as e:
            logger.error(f"Error during movie retrieval: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    ```

    - Define the `UserQuery` model for input validation.
    - Implement the `/movies/` endpoint to handle POST requests for movie recommendations.

### Streamlit Frontend (`app.py`)
1. **Imports and Logging Configuration**:
    ```python
    import streamlit as st
    import requests
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ```

    - Import necessary libraries and configure logging.

2. **Streamlit App Configuration**:
    ```python
    st.title("Movie Recommendation Chatbot")

    BASE_URL = "http://localhost:8000"  # Make sure this matches your FastAPI server port

    query = st.text_input("Anything about movies")

    if st.button("Filming it!"):
        try:
            response = requests.post(f"{BASE_URL}/movies/", json={"query": query})
            response.raise_for_status()
            if response.status_code == 200:
                movies = response.json().get("movies")
                st.write("Movies:")
                for movie in movies:
                    st.write(f"* {movie['title']} ({movie['genre']}) - {movie['plot']}")
            else:
                logger.error(f"Failed to fetch movies. Status code: {response.status_code}, Response: {response.text}")
                st.error("Failed to fetch movies. Please try again.")
        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException: {e}")
            st.error("Unable to connect to the movie recommendation service. Please try again later.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            st.error("An unexpected error occurred. Please try again later.")
    ```

    - Set up the Streamlit app title and base URL for the FastAPI server.
    - Create an input field for user queries and a button to trigger the recommendation request.
    - Handle the response from the FastAPI server and display the recommended movies.

## Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/movie-recommendation-chatbot.git
    cd movie-recommendation-chatbot
    ```

2. **Install Dependencies**:
    fastapi
    openai
    dotenv
    chromadb

## Running the App
1. **Start the FastAPI Server**:
    ```bash
    uvicorn main:app --reload
    ```

2. **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

3. **Access the App**:
    Open your browser and navigate to `http://localhost:8501` to interact with the Movie Recommendation Chatbot.

## Conclusion
This project demonstrates the integration of various technologies to build a personalized movie recommendation chatbot. It combines FastAPI for backend processing, Streamlit for frontend interaction, and leverages powerful language models and embeddings for data retrieval.
