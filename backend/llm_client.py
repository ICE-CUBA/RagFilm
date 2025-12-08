"""Ollama LLM Client for Movie RAG System."""

import os
from typing import List, Dict, Optional

import requests


class OllamaClient:
    """Client for interacting with Ollama LLM."""

    def __init__(
        self,
        base_url: str = 'http://localhost:11434',
        model: str = 'llama3'
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            model: Model name to use
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 800
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running."
        except requests.exceptions.Timeout:
            return "Error: Request to Ollama timed out."
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def create_recommendation(
        self,
        query: str,
        movies: List[Dict]
    ) -> str:
        """
        Create movie recommendations based on retrieved movies.

        Args:
            query: User's search query
            movies: List of retrieved movie dictionaries

        Returns:
            Recommendation text with citations
        """
        system_prompt = """You are a movie recommendation assistant.

RULES:
1. ONLY recommend movies from the provided list
2. NEVER make up movie titles or details
3. Include Wikipedia citation for each movie
4. Explain why each movie matches the query
5. Be concise but informative"""

        # Build context from movies
        context_parts = []
        for i, movie in enumerate(movies, 1):
            wiki_url = movie.get('Wiki Page', '')
            context_parts.append(f"""
Movie {i}:
- Title: {movie.get('Title', 'Unknown')}
- Year: {movie.get('Release Year', 'Unknown')}
- Genre: {movie.get('Genre', 'Unknown')}
- Director: {movie.get('Director', 'Unknown')}
- Wikipedia: {wiki_url}
- Plot excerpt: {movie.get('matched_chunk', '')[:500]}...
""")

        context = "\n".join(context_parts)

        user_prompt = f"""Query: {query}

Movies found:
{context}

Based on the query, provide 3-5 recommendations from the movies above.

For each recommendation:
1. Explain why it matches the query
2. Include the Wikipedia link

Format each recommendation as:
**Movie Title (Year)** - Brief explanation of why it matches
[Wikipedia](URL)

Only recommend movies from the list above. Do not make up any movies."""

        return self.generate(user_prompt, system_prompt)

    def health_check(self) -> bool:
        """
        Check if Ollama is available.

        Returns:
            True if Ollama is responding, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception:
            return []

