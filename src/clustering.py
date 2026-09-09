"""
Funções de clusterização (aprendizado não supervisionado) para explorar
tendências de rendimento de safra, conforme pedido na Entrega 1 da Fase 5.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def escalonar_features(df: pd.DataFrame, colunas: list) -> Tuple[np.ndarray, StandardScaler]:
    """
    Padroniza (z-score) as colunas informadas, retornando o array escalonado
    e o scaler ajustado (para eventual uso posterior).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[colunas])
    return X_scaled, scaler


def encontrar_k_ideal(
    X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> pd.DataFrame:
    """
    Testa diferentes valores de k (número de clusters) usando o método do
    cotovelo (inércia) e o coeficiente de silhueta, retornando uma tabela
    comparativa para apoiar a escolha do k ideal.

    Args:
        X_scaled: Matriz de features já escalonada
        k_min: Menor valor de k a testar
        k_max: Maior valor de k a testar (inclusive)
        random_state: Semente para reprodutibilidade

    Returns:
        DataFrame com colunas: k, inertia, silhouette_score
    """
    resultados = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        resultados.append(
            {"k": k, "inertia": kmeans.inertia_, "silhouette_score": silhouette}
        )
    return pd.DataFrame(resultados)


def clusterizar(
    X_scaled: np.ndarray, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Aplica o KMeans com o número de clusters definido, retornando os rótulos
    de cluster de cada observação e o modelo ajustado.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


def perfil_por_cluster(df: pd.DataFrame, labels: np.ndarray, colunas: list) -> pd.DataFrame:
    """
    Gera o perfil médio (por coluna) de cada cluster, além da contagem de
    observações — útil para interpretar o que cada cluster representa em
    termos de tendência de rendimento.
    """
    df_com_cluster = df.copy()
    df_com_cluster["cluster"] = labels

    perfil = df_com_cluster.groupby("cluster")[colunas].mean()
    perfil["total_observacoes"] = df_com_cluster.groupby("cluster").size()

    return perfil.reset_index()
