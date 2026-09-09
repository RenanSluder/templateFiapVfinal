"""
Parâmetros e caminhos do projeto FarmTech Solutions — Fase 5.

Este módulo concentra ajustes reutilizados pelo notebook, pelos scripts
auxiliares e pela documentação da estimativa de custos na AWS.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "src" / "data" / "crop_yield.csv"
GRAFICOS_DIR = ROOT_DIR / "assets" / "graficos"
REQUIREMENTS_PATH = ROOT_DIR / "requirements.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.2
KMEANS_K_MIN = 2
KMEANS_K_MAX = 10
N_CLUSTERS = 4

AWS_INSTANCE = "t3.micro"
AWS_STORAGE_GB = 50
AWS_HOURS_PER_MONTH = 730

AWS_COSTS = {
    "us-east-1": {
        "region_name": "US East (N. Virginia)",
        "ec2_hourly": 0.0104,
        "ebs_gb_month": 0.0800,
    },
    "sa-east-1": {
        "region_name": "South America (São Paulo)",
        "ec2_hourly": 0.0168,
        "ebs_gb_month": 0.1520,
    },
}

AWS_RECOMMENDED_REGION = "sa-east-1"
