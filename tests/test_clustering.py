"""Testes do módulo src/clustering.py"""

import numpy as np

from src.clustering import (
    clusterizar,
    encontrar_k_ideal,
    escalonar_features,
    perfil_por_cluster,
)
from src.data_processing import NUMERIC_COLUMNS, TARGET_COLUMN


def test_escalonar_features_produz_media_zero_desvio_um(dataset_exemplo):
    colunas = [*NUMERIC_COLUMNS, TARGET_COLUMN]
    X_scaled, scaler = escalonar_features(dataset_exemplo, colunas)

    assert X_scaled.shape == (len(dataset_exemplo), len(colunas))
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-6)


def test_encontrar_k_ideal_retorna_tabela_para_cada_k(dataset_exemplo):
    colunas = [*NUMERIC_COLUMNS, TARGET_COLUMN]
    X_scaled, _ = escalonar_features(dataset_exemplo, colunas)

    tabela = encontrar_k_ideal(X_scaled, k_min=2, k_max=4)

    assert list(tabela["k"]) == [2, 3, 4]
    assert (tabela["inertia"] > 0).all()
    assert tabela["silhouette_score"].between(-1, 1).all()


def test_clusterizar_retorna_rotulos_para_todas_observacoes(dataset_exemplo):
    colunas = [*NUMERIC_COLUMNS, TARGET_COLUMN]
    X_scaled, _ = escalonar_features(dataset_exemplo, colunas)

    labels, modelo = clusterizar(X_scaled, n_clusters=3)

    assert len(labels) == len(dataset_exemplo)
    assert set(labels) <= {0, 1, 2}
    assert modelo.n_clusters == 3


def test_perfil_por_cluster_tem_uma_linha_por_cluster(dataset_exemplo):
    colunas = [*NUMERIC_COLUMNS, TARGET_COLUMN]
    X_scaled, _ = escalonar_features(dataset_exemplo, colunas)
    labels, _ = clusterizar(X_scaled, n_clusters=3)

    perfil = perfil_por_cluster(dataset_exemplo, labels, colunas)

    assert len(perfil) == len(set(labels))
    assert "total_observacoes" in perfil.columns
    assert perfil["total_observacoes"].sum() == len(dataset_exemplo)
