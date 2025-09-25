import pandas as pd
from typing import List, Dict

def load_movie_data(path: str, sample_size: int = None) -> List[Dict]:
    """Load movie plot dataset from CSV file."""
    df = pd.read_csv(path, names=[
        "release_year", "title", "origin_ethnicity", "director",
        "cast", "genre", "wiki_page", "plot"
    ], header=0)  # Use header=0 to skip the first row (existing headers)

    # Select only rows with non-null plots
    df = df.dropna(subset=["plot"])
    
    # Sample only specific number of rows if specified
    if sample_size:
        df = df.head(sample_size)

    data = df[["title", "plot", "release_year", "director", "cast", "origin_ethnicity", "wiki_page", "genre"]].to_dict(orient="records")
    return data

if __name__ == "__main__":
    movies = load_movie_data("data/wiki_movie_plots_deduped.csv", sample_size=100)
    print(f"Loaded {len(movies)} movies")
    print(movies[:3])  # show first 3 entries