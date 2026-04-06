import pandas as pd
import numpy as np

def load_and_clean_data(path):
    df = pd.read_csv(path, on_bad_lines='skip')
    
    df.columns = df.columns.str.strip().str.lower()
    
    def parse_publication_date(date_str):
        """Parse date with multiple formats"""
        if pd.isna(date_str):
            return pd.NaT
        try:
            return pd.to_datetime(date_str, format='%m/%d/%Y')
        except:
            try:
                return pd.to_datetime(date_str, errors='coerce')
            except:
                return pd.NaT
    
    df['publication_date'] = df['publication_date'].apply(parse_publication_date)
    
    df['publication_year'] = df['publication_date'].dt.year
    
    df['bookid'] = df['bookid'].astype(int)
    df['ratings_count'] = df['ratings_count'].astype(int)
    df['text_reviews_count'] = df['text_reviews_count'].astype(int)
    df['num_pages'] = df['num_pages'].astype(int)
    
    df['language_code'] = df['language_code'].str.lower().str.strip()
    language_mapping = {
        'en-us': 'eng',
        'en-gb': 'eng',
        'en-au': 'eng',
        'pt-br': 'por',
    }
    df['language_code'] = df['language_code'].map(lambda x: language_mapping.get(x, x))
    
    df['title'] = df['title'].str.strip()
    df['authors'] = df['authors'].str.strip()
    df['publisher'] = df['publisher'].str.strip()
    
    df['average_rating'] = df['average_rating'].fillna(0)
    df['ratings_count'] = df['ratings_count'].fillna(0)
    df['text_reviews_count'] = df['text_reviews_count'].fillna(0)
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
    
    invalid_ratings = df[(df['average_rating'] < 0) | (df['average_rating'] > 5)]
    if len(invalid_ratings) > 0:
        df = df[(df['average_rating'] >= 0) & (df['average_rating'] <= 5)]
    
    negative_pages = df[df['num_pages'] < 0]
    if len(negative_pages) > 0:
        df = df[df['num_pages'] >= 0]
    
    df["popularity_score"] = df["average_rating"] * df["ratings_count"]
    df["engagement_score"] = df["ratings_count"] * df["average_rating"]
    
    df["rating_category"] = pd.cut(
        df["average_rating"],
        bins=[0, 3, 4, 4.5, 5],
        labels=["Faible", "Bon", "Très Bon", "Excellent"],
        include_lowest=True
    )
    
    df["length_category"] = pd.cut(
        df["num_pages"],
        bins=[0, 200, 400, 700, 2000],
        labels=["Court", "Moyen", "Long", "Très Long"],
        include_lowest=True
    )
    
    df["isbn_clean"] = df["isbn13"].fillna(df["isbn"]).astype(str).str.strip()
    df.loc[df["isbn_clean"].isin(["nan", "", "None", "none"]), "isbn_clean"] = np.nan
    df["image_url"] = np.where(
        df["isbn_clean"].notna(),
        "https://covers.openlibrary.org/b/isbn/" + df["isbn_clean"] + "-L.jpg",
        "assets/placeholder.png"
    )
    
    return df
