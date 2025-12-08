"""Script to clean the Wikipedia Movie Plots dataset."""

import argparse
import os

import pandas as pd


def clean_movie_data(input_path: str, output_path: str, min_plot_length: int = 50):
    """
    Clean the movie dataset.

    Args:
        input_path: Path to raw CSV file
        output_path: Path to save cleaned CSV
        min_plot_length: Minimum plot length to keep
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Original dataset size: {len(df)} movies")
    
    # Display column names
    print(f"Columns: {list(df.columns)}")
    
    # Remove rows with missing or short plots
    df = df.dropna(subset=['Plot'])
    df['Plot'] = df['Plot'].astype(str)
    df = df[df['Plot'].str.len() >= min_plot_length]
    
    print(f"After removing short plots: {len(df)} movies")
    
    # Clean title
    if 'Title' in df.columns:
        df['Title'] = df['Title'].fillna('Unknown')
        df['Title'] = df['Title'].str.strip()
    
    # Clean genre
    if 'Genre' in df.columns:
        df['Genre'] = df['Genre'].fillna('Unknown')
        df['Genre'] = df['Genre'].str.strip()
    
    # Clean director
    if 'Director' in df.columns:
        df['Director'] = df['Director'].fillna('Unknown')
        df['Director'] = df['Director'].str.strip()
    
    # Clean year
    if 'Release Year' in df.columns:
        df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce')
        df['Release Year'] = df['Release Year'].fillna(0).astype(int)
    
    # Clean wiki page
    if 'Wiki Page' in df.columns:
        df['Wiki Page'] = df['Wiki Page'].fillna('')
    
    # Remove duplicates by title
    df = df.drop_duplicates(subset=['Title'], keep='first')
    print(f"After removing duplicates: {len(df)} movies")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total movies: {len(df)}")
    print(f"  Year range: {df['Release Year'].min()} - {df['Release Year'].max()}")
    print(f"  Avg plot length: {df['Plot'].str.len().mean():.0f} characters")
    
    if 'Genre' in df.columns:
        print(f"\nTop 10 genres:")
        genre_counts = df['Genre'].value_counts().head(10)
        for genre, count in genre_counts.items():
            print(f"    {genre}: {count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean movie dataset")
    parser.add_argument(
        '--input',
        type=str,
        default='data/wiki_movie_plots_deduped.csv',
        help='Path to input CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/movies_cleaned.csv',
        help='Path to output CSV'
    )
    parser.add_argument(
        '--min-plot-length',
        type=int,
        default=50,
        help='Minimum plot length'
    )
    
    args = parser.parse_args()
    clean_movie_data(args.input, args.output, args.min_plot_length)


if __name__ == "__main__":
    main()

