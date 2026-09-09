"""
Funções de carregamento, limpeza e pré-processamento do dataset
`crop_yield.csv` (Fase 5 - Cap 1 - FarmTech Solutions).

Colunas esperadas:
    Crop, Precipitation (mm day), Specific Humidity at 2 Meters (g/kg),
    Relative Humidity at 2 Meters (%), Temperature at 2 Meters (C), Yield
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

NUMERIC_COLUMNS: List[str] = [
    "Precipitation (mm day)",
    "Specific Humidity at 2 Meters (g/kg)",
    "Relative Humidity at 2 Meters (%)",
    "Temperature at 2 Meters (C)",
]
TARGET_COLUMN = "Yield"
CATEGORICAL_COLUMN = "Crop"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Carrega o dataset de rendimento de safra a partir de um arquivo CSV.

    Args:
        csv_path: Caminho para o arquivo crop_yield.csv

    Returns:
        DataFrame com os dados carregados

    Raises:
        FileNotFoundError: se o arquivo não existir
        ValueError: se alguma coluna esperada estiver ausente
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {csv_path}")

    df = pd.read_csv(csv_path)

    esperadas = {CATEGORICAL_COLUMN, TARGET_COLUMN, *NUMERIC_COLUMNS}
    faltantes = esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes no dataset: {faltantes}")

    return df


def detectar_outliers_iqr(df: pd.DataFrame, coluna: str, k: float = 1.5) -> pd.Series:
    """
    Detecta outliers em uma coluna numérica usando o método IQR (intervalo
    interquartil), retornando uma máscara booleana (True = é outlier).

    Args:
        df: DataFrame com os dados
        coluna: Nome da coluna numérica a avaliar
        k: Multiplicador do IQR (1.5 é o valor clássico de Tukey)

    Returns:
        Series booleana, mesmo índice do DataFrame, True onde há outlier
    """
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - k * iqr
    limite_superior = q3 + k * iqr
    return (df[coluna] < limite_inferior) | (df[coluna] > limite_superior)


def resumo_outliers(df: pd.DataFrame, colunas: List[str] | None = None) -> pd.DataFrame:
    """
    Gera um resumo da quantidade e percentual de outliers (método IQR) para
    cada coluna numérica informada.

    Args:
        df: DataFrame com os dados
        colunas: Lista de colunas a avaliar. Se None, usa NUMERIC_COLUMNS + TARGET_COLUMN

    Returns:
        DataFrame com colunas: coluna, total_outliers, percentual_outliers
    """
    colunas = colunas or [*NUMERIC_COLUMNS, TARGET_COLUMN]
    registros = []
    for coluna in colunas:
        mascara = detectar_outliers_iqr(df, coluna)
        registros.append(
            {
                "coluna": coluna,
                "total_outliers": int(mascara.sum()),
                "percentual_outliers": round(100 * mascara.mean(), 2),
            }
        )
    return pd.DataFrame(registros)


def preparar_features_target(
    df: pd.DataFrame,
    one_hot_crop: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features (X) e alvo (y) para a modelagem de regressão supervisionada,
    aplicando one-hot encoding na coluna categórica `Crop` quando solicitado.

    Args:
        df: DataFrame completo (já limpo)
        one_hot_crop: Se True, expande a coluna Crop em dummies (0/1)

    Returns:
        Tupla (X, y) prontas para split treino/teste
    """
    y = df[TARGET_COLUMN].copy()

    if one_hot_crop:
        X = pd.get_dummies(df.drop(columns=[TARGET_COLUMN]), columns=[CATEGORICAL_COLUMN])
    else:
        X = df[NUMERIC_COLUMNS].copy()

    return X, y


def matriz_correlacao_numerica(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a matriz de correlação de Pearson das colunas numéricas + alvo"""
    colunas = [*NUMERIC_COLUMNS, TARGET_COLUMN]
    return df[colunas].corr()
