"""Testes do módulo src/regression_models.py"""

import numpy as np

from src.data_processing import preparar_features_target
from src.regression_models import (
    avaliar_modelo,
    construir_modelos,
    dividir_treino_teste,
    escalonar_treino_teste,
    treinar_e_avaliar_todos,
)


def test_dividir_treino_teste_respeita_proporcao(dataset_exemplo):
    X, y = preparar_features_target(dataset_exemplo)
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y, test_size=0.2)

    total = len(X)
    assert len(X_test) == round(total * 0.2)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_escalonar_treino_teste_mantem_shape(dataset_exemplo):
    X, y = preparar_features_target(dataset_exemplo)
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)

    X_train_scaled, X_test_scaled, scaler = escalonar_treino_teste(X_train, X_test)

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


def test_construir_modelos_retorna_cinco_algoritmos_distintos():
    modelos = construir_modelos()

    assert len(modelos) == 5
    nomes_classes = {type(m).__name__ for m in modelos.values()}
    assert len(nomes_classes) == 5  # todos os algoritmos são de classes diferentes


def test_avaliar_modelo_retorna_metricas_esperadas():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    metricas = avaliar_modelo(y_true, y_pred)

    assert set(metricas.keys()) == {"MAE", "MSE", "RMSE", "R2"}
    assert metricas["MAE"] >= 0
    assert metricas["MSE"] >= 0
    assert metricas["RMSE"] >= 0
    assert np.isclose(metricas["RMSE"], np.sqrt(metricas["MSE"]))


def test_avaliar_modelo_previsao_perfeita_da_metricas_zero_e_r2_um():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    metricas = avaliar_modelo(y_true, y_pred)

    assert metricas["MAE"] == 0
    assert metricas["MSE"] == 0
    assert metricas["R2"] == 1.0


def test_treinar_e_avaliar_todos_retorna_tabela_ordenada_por_r2(dataset_exemplo):
    X, y = preparar_features_target(dataset_exemplo)
    X_train, X_test, y_train, y_test = dividir_treino_teste(X, y)
    X_train_scaled, X_test_scaled, _ = escalonar_treino_teste(X_train, X_test)

    modelos = construir_modelos()
    tabela = treinar_e_avaliar_todos(modelos, X_train_scaled, X_test_scaled, y_train, y_test)

    assert len(tabela) == 5
    assert list(tabela.columns) == ["Modelo", "MAE", "MSE", "RMSE", "R2"]
    # tabela deve estar ordenada do maior para o menor R2
    assert list(tabela["R2"]) == sorted(tabela["R2"], reverse=True)
