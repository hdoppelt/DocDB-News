import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_search_index(df: pd.DataFrame):
    """Build and return a TF-IDF vectorizer and matrix for the dataset."""

    if df.empty:
        return None, None

    search_text = (
        df["title"].fillna("").astype(str) + " "
        + df["text"].fillna("").astype(str) + " "
        + df["summary"].fillna("").astype(str)
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(search_text)
    return vectorizer, doc_matrix


def search_documents(df: pd.DataFrame, query: str, vectorizer=None, doc_matrix=None) -> pd.DataFrame:
    """Return all matching documents for a query using TF-IDF, sorted by relevance."""

    if df.empty or not query.strip():
        return df.head(0).copy()

    if vectorizer is None or doc_matrix is None:
        vectorizer, doc_matrix = build_search_index(df)

    if vectorizer is None or doc_matrix is None:
        return df.head(0).copy()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, doc_matrix).flatten()

    results = df.copy()
    results["score"] = scores
    results = results[results["score"] > 0.05]
    results = results.sort_values("score", ascending=False)

    return results


def get_related_documents(df: pd.DataFrame, selected_index: int, vectorizer=None, doc_matrix=None, top_k: int = 5) -> pd.DataFrame:
    """Return the most similar documents to the selected document."""

    if df.empty or selected_index not in df.index:
        return df.head(0).copy()

    if vectorizer is None or doc_matrix is None:
        vectorizer, doc_matrix = build_search_index(df)

    if vectorizer is None or doc_matrix is None:
        return df.head(0).copy()

    selected_row_position = df.index.get_loc(selected_index)
    selected_vector = doc_matrix[selected_row_position]
    similarity_scores = cosine_similarity(selected_vector, doc_matrix).flatten()

    results = df.copy()
    results["similarity"] = similarity_scores
    results = results.drop(index=selected_index, errors="ignore")
    results = results.sort_values("similarity", ascending=False)

    return results.head(top_k)
