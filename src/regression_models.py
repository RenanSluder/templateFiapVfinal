"""
Módulo de modelagem preditiva: treina e avalia 5 algoritmos de regressão
supervisionada distintos para prever o rendimento (Yield) da safra,
conforme pedido na Entrega 1 da Fase 5.
"""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42


def dividir_treino_teste(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide os dados em treino (80%) e teste (20%) por padrão"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def escalonar_treino_teste(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Ajusta o StandardScaler no treino e aplica em treino e teste"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def construir_modelos(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Constrói o dicionário com os 5 modelos de regressão (algoritmos distintos)
    exigidos pela atividade: Linear Regression, Ridge, Random Forest,
    Gradient Boosting e Support Vector Regression (SVR).
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4, random_state=random_state
        ),
        "Support Vector Regression": SVR(kernel="rbf", C=10, epsilon=0.1),
    }


def avaliar_modelo(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula as métricas de avaliação pertinentes a um problema de regressão:
    MAE, MSE, RMSE e R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def treinar_e_avaliar_todos(
    modelos: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Treina cada modelo do dicionário e avalia no conjunto de teste,
    retornando uma tabela comparativa ordenada pelo melhor R².

    Returns:
        DataFrame com colunas: Modelo, MAE, MSE, RMSE, R2
    """
    linhas = []
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        metricas = avaliar_modelo(y_test.values, y_pred)
        linhas.append({"Modelo": nome, **metricas})

    tabela = pd.DataFrame(linhas).sort_values("R2", ascending=False).reset_index(drop=True)
    return tabela
