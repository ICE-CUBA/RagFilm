"""Script to build FAISS index from movie dataset."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.vector_store import MovieVectorStore


def main():
    """Main function to build the index."""
    parser = argparse.ArgumentParser(description="Build FAISS index for movies")
    parser.add_argument(
        '--input',
        type=str,
        default='data/movies_cleaned.csv',
        help='Path to cleaned CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/movie_index',
        help='Output path prefix for index files'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=10,
        help='Number of sentences per chunk'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Please run clean_data.py first or download the dataset.")
        sys.exit(1)
    
    # Initialize vector store
    print(f"Initializing vector store with model: {args.model}")
    vector_store = MovieVectorStore(model_name=args.model)
    
    # Build index
    print(f"Building index from: {args.input}")
    print(f"Batch size: {args.batch_size}")
    print(f"Chunk size: {args.chunk_size} sentences")
    
    vector_store.build_index(
        csv_path=args.input,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        save_path=args.output
    )
    
    print("\nIndex build complete!")
    print(f"Index file: {args.output}.faiss")
    print(f"Metadata file: {args.output}_metadata.pkl")
    
    # Verify by loading
    print("\nVerifying index...")
    vector_store2 = MovieVectorStore(model_name=args.model)
    vector_store2.load_index(
        f"{args.output}.faiss",
        f"{args.output}_metadata.pkl"
    )
    
    print(f"Unique movies: {vector_store2.get_unique_movies_count()}")
    
    # Test search
    print("\nTesting search with query: 'sci-fi movie about space'")
    results = vector_store2.search("sci-fi movie about space", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['Title']} ({r['Release Year']}) - Score: {r['similarity_score']:.3f}")


if __name__ == "__main__":
    main()

