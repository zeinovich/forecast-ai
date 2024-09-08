import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def auto_determine_clusters(df: pd.DataFrame, max_clusters: int = 10) -> int:
    """
    Автоматически определяем количество кластеров на основе метрики (например, Silhouette Score).
    """
    best_score = -1
    best_n_clusters = 2

    for n_clusters in range(2, max_clusters + 1):
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(df)
        score = silhouette_score(df, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    return best_n_clusters
