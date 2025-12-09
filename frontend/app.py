"""Streamlit Frontend for Movie RAG System - Chat Interface."""

import os
import requests
import streamlit as st
from datetime import datetime


# Configuration
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

# Sample queries for demo
SAMPLE_QUERIES = [
    "🚀 Sci-fi movies about space exploration",
    "💕 Romantic comedies from the 90s",
    "🎬 Christopher Nolan's best thriller films",
    "🦸 Superhero movies with complex villains",
]


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
    """Search for movies using the backend API."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Search failed: {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to backend. Please make sure the server is running."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def get_genre_emoji(genre: str) -> str:
    """Return emoji based on movie genre."""
    genre_lower = genre.lower() if genre else ""
    emoji_map = {
        "action": "💥", "adventure": "🗺️", "animation": "🎨", "comedy": "😂",
        "crime": "🔫", "documentary": "📹", "drama": "🎭", "family": "👨‍👩‍👧",
        "fantasy": "🧙", "history": "📜", "horror": "👻", "music": "🎵",
        "mystery": "🔍", "romance": "💕", "science fiction": "🚀", "sci-fi": "🚀",
        "thriller": "😰", "war": "⚔️", "western": "🤠",
    }
    for key, emoji in emoji_map.items():
        if key in genre_lower:
            return emoji
    return "🎬"


def format_movie_response(result: dict) -> str:
    """Format the movie search result as a nice message."""
    if "error" in result:
        return f"❌ **Error:** {result['error']}"
    
    answer = result.get("answer", "")
    movies = result.get("movies", [])
    
    response_parts = []
    
    # Add the AI answer
    if answer:
        response_parts.append(f"💭 **Here's what I found:**\n\n{answer}")
    
    # Add movie recommendations
    if movies:
        response_parts.append("\n\n---\n\n🎬 **Top Recommendations:**\n")
        for i, movie in enumerate(movies, 1):
            title = movie.get("Title", "Unknown")
            year = movie.get("Release Year", "N/A")
            genre = movie.get("Genre", "Unknown")
            director = movie.get("Director", "Unknown")
            score = movie.get("similarity_score", 0)
            wiki_url = movie.get("Wiki Page", "")
            
            emoji = get_genre_emoji(genre)
            
            movie_info = f"""
**{i}. {emoji} {title}** ({year})
- 🎬 Genre: {genre}
- 🎥 Director: {director}
- 📊 Match: {score:.1%}"""
            
            if wiki_url:
                movie_info += f"\n- 🔗 [Wikipedia]({wiki_url})"
            
            response_parts.append(movie_info)
    
    return "\n".join(response_parts)


def render_movie_cards(movies: list):
    """Render movie cards in the chat."""
    if not movies:
        return
    
    for i, movie in enumerate(movies, 1):
        title = movie.get("Title", "Unknown")
        year = movie.get("Release Year", "N/A")
        genre = movie.get("Genre", "Unknown")
        director = movie.get("Director", "Unknown")
        score = movie.get("similarity_score", 0)
        wiki_url = movie.get("Wiki Page", "")
        matched_chunk = movie.get("matched_chunk", "")
        
        emoji = get_genre_emoji(genre)
        
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col1:
            st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{emoji}</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**{i}. {title}** ({year})")
            st.caption(f"🎬 {genre} · 🎥 {director}")
        
        with col3:
            st.metric("Match", f"{score:.0%}")
        
        # Expandable plot
        if matched_chunk:
            with st.expander("📖 Plot excerpt"):
                st.caption(matched_chunk)
        
        # Wiki link
        if wiki_url:
            st.markdown(f"[🔗 View on Wikipedia]({wiki_url})")
        
        st.divider()


def main():
    """Main application."""
    # Page config
    st.set_page_config(
        page_title="🎬 CineRAG - AI Movie Assistant",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi! I'm your AI movie assistant. Ask me anything about movies!\n\nFor example:\n- *\"Find me sci-fi movies about time travel\"*\n- *\"What are some good comedies from the 90s?\"*\n- *\"Recommend movies directed by Christopher Nolan\"*",
                "movies": None
            }
        ]
    
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5

    # Custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700&display=swap');
        
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
        
        /* Header */
        .main-header {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 0.5rem;
        }
        
        .main-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #0891b2 0%, #0d9488 50%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0;
        }
        
        .subtitle {
            font-size: 0.95rem;
            color: #64748b;
        }
        
        /* Chat styling */
        .stChatMessage {
            background: white !important;
            border-radius: 12px !important;
            border: 1px solid #e2e8f0 !important;
            margin-bottom: 1rem !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
        }
        
        /* Status indicators */
        .status-online {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.3);
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        
        .status-offline {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        
        /* Chat input */
        .stChatInput > div {
            background: white !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 12px !important;
        }
        
        .stChatInput > div:focus-within {
            border-color: #0891b2 !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #0891b2 0%, #0d9488 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(8, 145, 178, 0.3) !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #0891b2 !important;
            font-size: 1.2rem !important;
        }
        
        /* Hide streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Divider */
        hr {
            border-color: #e2e8f0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # System Status
        st.markdown("### 📡 Status")
        health = check_backend_health()
        
        if health:
            st.markdown("""
            <div class="status-online">
                <span style="color: #16a34a;">● Backend Online</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎥 Movies", f"{health.get('unique_movies', 0):,}")
            with col2:
                st.metric("📊 Vectors", f"{health.get('total_vectors', 0):,}")
            
            if health.get('ollama_available'):
                st.markdown("""
                <div class="status-online">
                    <span style="color: #16a34a;">● LLM Ready</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-offline">
                    <span style="color: #d97706;">● LLM Unavailable</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-offline">
                <span style="color: #dc2626;">● Backend Offline</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Settings
        st.markdown("### ⚙️ Settings")
        st.session_state.top_k = st.slider(
            "Results to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of movies to retrieve"
        )
        
        st.divider()
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Chat cleared! Ask me anything about movies!",
                    "movies": None
                }
            ]
            st.rerun()
        
        st.divider()
        
        # Sample Queries
        st.markdown("### 💡 Try These")
        for query in SAMPLE_QUERIES:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                # Add to messages and trigger search
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })
                st.rerun()
        
        st.divider()
        st.caption("🔍 FAISS · 🤖 MiniLM · 🦙 Llama 3")

    # Main Chat Area
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎬 CineRAG</h1>
        <p class="subtitle">Your AI-Powered Movie Discovery Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🎬" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])
                
                # Show movie cards if available
                if message.get("movies"):
                    st.divider()
                    render_movie_cards(message["movies"])

    # Chat input
    if prompt := st.chat_input("Ask me about movies... (e.g., 'Find sci-fi movies about AI')"):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant", avatar="🎬"):
            with st.spinner("🔍 Searching for movies..."):
                result = search_movies(prompt, st.session_state.top_k)
            
            if "error" in result:
                response = f"❌ **Oops!** {result['error']}"
                movies = None
            else:
                answer = result.get("answer", "Here are some movies I found for you:")
                movies = result.get("movies", [])
                
                response = f"💭 {answer}"
                
                st.markdown(response)
                
                if movies:
                    st.divider()
                    st.markdown("**🎬 Top Recommendations:**")
                    render_movie_cards(movies)
                
                # Add citations
                citations = result.get("citations", [])
                if citations:
                    with st.expander("📚 Sources"):
                        for c in citations:
                            if c.get('wiki_url'):
                                st.markdown(f"- [{c['title']} ({c['year']})]({c['wiki_url']})")
        
        # Save assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "movies": movies if 'movies' in dir() else None
        })
        
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem;">
        <strong style="color: #0891b2;">CineRAG</strong> · AI-Powered Movie Discovery · NLP Course Project © 2024
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
