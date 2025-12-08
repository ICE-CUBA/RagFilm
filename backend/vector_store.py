"""FAISS Vector Store for Movie RAG System."""

import os
import pickle
from typing import List, Dict, Optional

import faiss
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class MovieVectorStore:
    """Vector store using FAISS for movie retrieval."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector store.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # MiniLM-L6-v2 dimension
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        
        # Load spaCy for sentence segmentation
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Increase max length for long plots
        self.nlp.max_length = 2000000

    def _chunk_text(self, text: str, chunk_size: int = 10, min_sentences: int = 3) -> List[str]:
        """
        Chunk text into segments of sentences.

        Args:
            text: Input text to chunk
            chunk_size: Number of sentences per chunk
            min_sentences: Minimum sentences for a valid chunk

        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) < 50:
            return []
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if len(sentences) < min_sentences:
            return [text] if len(text) >= 50 else []
        
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            if len(chunk_sentences) >= min_sentences or i == 0:
                chunks.append(' '.join(chunk_sentences))
        
        return chunks

    def _create_metadata_prefix(self, row: pd.Series) -> str:
        """
        Create metadata prefix for a movie.

        Args:
            row: DataFrame row with movie data

        Returns:
            Formatted metadata string
        """
        prefix = f"Title: {row.get('Title', 'Unknown')}\n"
        prefix += f"Genre: {row.get('Genre', 'Unknown')}\n"
        prefix += f"Director: {row.get('Director', 'Unknown')}\n"
        prefix += f"Year: {row.get('Release Year', 'Unknown')}\n"
        prefix += f"Origin: {row.get('Origin/Ethnicity', 'Unknown')}\n"
        return prefix

    def build_index(
        self,
        csv_path: str,
        batch_size: int = 32,
        chunk_size: int = 10,
        save_path: str = 'data/movie_index'
    ) -> None:
        """
        Build FAISS index from movie CSV.

        Args:
            csv_path: Path to the cleaned CSV file
            batch_size: Batch size for encoding
            chunk_size: Number of sentences per chunk
            save_path: Path prefix to save index and metadata
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} movies")

        all_texts = []
        self.metadata = []

        print("Processing movies and creating chunks...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            plot = str(row.get('Plot', ''))
            chunks = self._chunk_text(plot, chunk_size=chunk_size)
            
            if not chunks:
                continue
            
            metadata_prefix = self._create_metadata_prefix(row)
            
            for chunk in chunks:
                # Combine metadata with chunk for embedding
                text_to_embed = f"{metadata_prefix}\nPlot: {chunk}"
                all_texts.append(text_to_embed)
                
                self.metadata.append({
                    'Title': row.get('Title', 'Unknown'),
                    'Genre': row.get('Genre', 'Unknown'),
                    'Director': row.get('Director', 'Unknown'),
                    'Release Year': int(row.get('Release Year', 0)) if pd.notna(row.get('Release Year')) else 0,
                    'Plot': plot,
                    'Wiki Page': row.get('Wiki Page', ''),
                    'Origin': row.get('Origin/Ethnicity', 'Unknown'),
                    'chunk': chunk,
                    'chunk_text': text_to_embed
                })

        print(f"Total chunks: {len(all_texts)}")
        print("Generating embeddings...")
        
        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding"):
            batch = all_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        
        # L2 normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
        
        # Save index and metadata
        self.save_index(save_path)

    def save_index(self, save_path: str) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            save_path: Path prefix for saving files
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        faiss_path = f"{save_path}.faiss"
        metadata_path = f"{save_path}_metadata.pkl"
        
        print(f"Saving index to {faiss_path}...")
        faiss.write_index(self.index, faiss_path)
        
        print(f"Saving metadata to {metadata_path}...")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print("Index saved successfully")

    def load_index(
        self,
        index_path: str = 'data/movie_index.faiss',
        metadata_path: str = 'data/movie_index_metadata.pkl'
    ) -> bool:
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file

        Returns:
            True if loading succeeded, False otherwise
        """
        if not os.path.exists(index_path):
            print(f"Index file not found: {index_path}")
            return False
        
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            return False
        
        print(f"Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.metadata)} metadata entries")
        
        return True

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for movies matching the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of movie dictionaries with matched chunks
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search with more results to handle deduplication
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Deduplicate by movie title, keeping best score
        seen_titles = set()
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx]
            title = meta['Title']
            
            if title in seen_titles:
                continue
            
            seen_titles.add(title)
            results.append({
                'Title': meta['Title'],
                'Genre': meta['Genre'],
                'Director': meta['Director'],
                'Release Year': meta['Release Year'],
                'Plot': meta['Plot'],
                'Wiki Page': meta['Wiki Page'],
                'similarity_score': float(score),
                'matched_chunk': meta['chunk']
            })
            
            if len(results) >= top_k:
                break
        
        return results

    def get_unique_movies_count(self) -> int:
        """Get count of unique movies in the index."""
        if not self.metadata:
            return 0
        titles = set(m['Title'] for m in self.metadata)
        return len(titles)

