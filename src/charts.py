from collections import Counter
import pandas as pd


def get_top_items(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Count comma-separated items in a Series and return the top ones."""
    
    counter = Counter()

    for value in series.dropna():
        items = [item.strip() for item in str(value).split(",") if item.strip()]
        counter.update(items)

    most_common = counter.most_common(top_n)
    return pd.DataFrame(most_common, columns=["item", "count"])
