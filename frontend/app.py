"""Streamlit Frontend for Movie RAG System."""

import os

import requests
import streamlit as st


# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')


def check_backend_health():
    """Check if backend is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def search_movies(query: str, top_k: int = 5):
    """
    Search for movies using the backend API.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        API response dict or None on error
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to backend. Please make sure the server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def main():
    """Main application."""
    # Page config
    st.set_page_config(
        page_title="Movie RAG System",
        page_icon="film_frames",
        layout="wide"
    )

    # Initialize session state
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'search_result' not in st.session_state:
        st.session_state.search_result = None

    # Custom CSS
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .movie-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #1f77b4;
        }
        .movie-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: #333;
        }
        .movie-meta {
            color: #666;
            font-size: 0.9rem;
        }
        .answer-box {
            background-color: #e8f4f8;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .score-badge {
            background-color: #28a745;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 5px;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-title">Movie RAG System</p>', unsafe_allow_html=True)
    st.markdown("Search for movies using natural language queries")

    # Sidebar for status
    with st.sidebar:
        st.header("System Status")
        health = check_backend_health()
        if health:
            st.success("Backend: Connected")
            st.info(f"Movies indexed: {health.get('unique_movies', 0):,}")
            st.info(f"Vectors: {health.get('total_vectors', 0):,}")
            if health.get('ollama_available'):
                st.success("LLM: Available")
            else:
                st.warning("LLM: Not available")
        else:
            st.error("Backend: Disconnected")

        st.divider()
        st.header("Settings")
        top_k = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=5,
            key="top_k_slider"
        )

    # Main search interface
    query = st.text_input(
        "Enter your search query",
        placeholder="Example: I want to watch a sci-fi movie about time travel",
        key="query_input"
    )

    search_clicked = st.button("Search", type="primary", key="search_button")

    # Process search
    if search_clicked and query.strip():
        with st.spinner("Searching..."):
            result = search_movies(query.strip(), top_k)
            st.session_state.search_result = result

    # Display results
    result = st.session_state.search_result
    if result:
        # Display LLM answer
        st.subheader("Recommendations")
        answer = result.get("answer", "No answer generated.")
        st.markdown(answer)

        # Display movie results
        st.subheader("Retrieved Movies")

        movies = result.get("movies", [])
        if movies:
            for i, movie in enumerate(movies):
                title = movie.get("Title", "Unknown")
                year = movie.get("Release Year", "N/A")
                genre = movie.get("Genre", "Unknown")
                director = movie.get("Director", "Unknown")
                score = movie.get("similarity_score", 0)
                wiki_url = movie.get("Wiki Page", "")

                st.markdown(f"### {i+1}. {title} ({year})")
                st.markdown(f"**Genre:** {genre} | **Director:** {director} | **Similarity:** {score:.3f}")

                if wiki_url:
                    st.markdown(f"[View on Wikipedia]({wiki_url})")

                with st.expander(f"View plot excerpt", expanded=False):
                    st.write(movie.get("matched_chunk", "No excerpt available"))

                st.divider()
        else:
            st.info("No movies found matching your query.")

        # Citations section
        citations = result.get("citations", [])
        if citations:
            with st.expander("All Citations"):
                for citation in citations:
                    wiki_url = citation.get('wiki_url', '')
                    if wiki_url:
                        st.markdown(
                            f"- [{citation['title']} ({citation['year']})]({wiki_url})"
                        )

    elif search_clicked:
        st.warning("Please enter a search query.")

    # Footer
    st.divider()
    st.caption("Movie RAG System - Powered by FAISS, MiniLM, and Llama 3")


if __name__ == "__main__":
    main()
