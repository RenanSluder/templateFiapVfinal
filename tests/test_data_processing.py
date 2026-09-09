"""Testes do módulo src/data_processing.py"""

import pandas as pd
import pytest

from src.data_processing import (
    CATEGORICAL_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    detectar_outliers_iqr,
    load_dataset,
    matriz_correlacao_numerica,
    preparar_features_target,
    resumo_outliers,
)


def test_load_dataset_carrega_csv_valido(tmp_path, dataset_exemplo):
    csv_path = tmp_path / "crop_yield.csv"
    dataset_exemplo.to_csv(csv_path, index=False)

    df = load_dataset(csv_path)

    assert len(df) == len(dataset_exemplo)
    assert CATEGORICAL_COLUMN in df.columns
    assert TARGET_COLUMN in df.columns


def test_load_dataset_arquivo_inexistente_gera_erro(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nao_existe.csv")


def test_load_dataset_colunas_faltantes_gera_erro(tmp_path):
    df_invalido = pd.DataFrame({"Crop": ["Rice"], "Yield": [3.0]})
    csv_path = tmp_path / "invalido.csv"
    df_invalido.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_dataset(csv_path)


def test_detectar_outliers_iqr_identifica_valor_extremo(dataset_exemplo):
    mascara = detectar_outliers_iqr(dataset_exemplo, TARGET_COLUMN)

    assert mascara.iloc[0] == True  # noqa: E712 (valor injetado como outlier)
    assert mascara.sum() >= 1


def test_resumo_outliers_retorna_estrutura_esperada(dataset_exemplo):
    resumo = resumo_outliers(dataset_exemplo)

    assert set(resumo.columns) == {"coluna", "total_outliers", "percentual_outliers"}
    assert len(resumo) == len(NUMERIC_COLUMNS) + 1  # + coluna Yield


def test_preparar_features_target_com_one_hot(dataset_exemplo):
    X, y = preparar_features_target(dataset_exemplo, one_hot_crop=True)

    assert len(X) == len(dataset_exemplo)
    assert len(y) == len(dataset_exemplo)
    assert TARGET_COLUMN not in X.columns
    # Colunas dummy do Crop devem existir (prefixo "Crop_")
    assert any(col.startswith("Crop_") for col in X.columns)


def test_preparar_features_target_sem_one_hot(dataset_exemplo):
    X, y = preparar_features_target(dataset_exemplo, one_hot_crop=False)

    assert list(X.columns) == NUMERIC_COLUMNS
    assert len(y) == len(dataset_exemplo)


def test_matriz_correlacao_numerica_e_quadrada_e_simetrica(dataset_exemplo):
    corr = matriz_correlacao_numerica(dataset_exemplo)

    assert corr.shape[0] == corr.shape[1]
    assert (corr.round(6) == corr.T.round(6)).all().all()
