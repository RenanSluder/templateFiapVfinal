"""Fixtures compartilhadas dos testes do projeto (Fase 5 - Cap 1)"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Garante que "src" seja importável durante os testes
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def dataset_exemplo() -> pd.DataFrame:
    """
    Pequeno dataset sintético de rendimento de safra, no mesmo formato do
    crop_yield.csv, usado para testar as funções de processamento sem
    depender do arquivo real em disco.
    """
    rng = np.random.default_rng(42)
    n = 60
    culturas = np.array(["Maize", "Rice", "Wheat"])

    df = pd.DataFrame(
        {
            "Crop": rng.choice(culturas, size=n),
            "Precipitation (mm day)": rng.uniform(40, 300, size=n),
            "Specific Humidity at 2 Meters (g/kg)": rng.uniform(5, 22, size=n),
            "Relative Humidity at 2 Meters (%)": rng.uniform(30, 90, size=n),
            "Temperature at 2 Meters (C)": rng.uniform(8, 34, size=n),
            "Yield": rng.uniform(1.0, 8.0, size=n),
        }
    )

    # Injeta um outlier proposital de rendimento para os testes de detecção
    df.loc[0, "Yield"] = 50.0

    return df
