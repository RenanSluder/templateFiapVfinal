#!/usr/bin/env python3
"""Calcula o custo mensal/anual On-Demand da API FarmTech na AWS."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import AWS_COSTS, AWS_HOURS_PER_MONTH, AWS_STORAGE_GB


def estimar(region_key: str) -> dict[str, float]:
    cfg = AWS_COSTS[region_key]
    ec2_month = cfg["ec2_hourly"] * AWS_HOURS_PER_MONTH
    ebs_month = cfg["ebs_gb_month"] * AWS_STORAGE_GB
    total_month = ec2_month + ebs_month
    return {
        "ec2_month": round(ec2_month, 2),
        "ebs_month": round(ebs_month, 2),
        "total_month": round(total_month, 2),
        "total_year": round(total_month * 12, 2),
    }


def main() -> None:
    print("Estimativa On-Demand (100%) — t3.micro + 50 GB EBS gp3\n")
    print(f"{'Região':<28} {'EC2/mês':>10} {'EBS/mês':>10} {'Total/mês':>10} {'Total/ano':>10}")
    for key, cfg in AWS_COSTS.items():
        valores = estimar(key)
        nome = f"{cfg['region_name']} ({key})"
        print(
            f"{nome:<28} "
            f"${valores['ec2_month']:>8.2f} "
            f"${valores['ebs_month']:>8.2f} "
            f"${valores['total_month']:>8.2f} "
            f"${valores['total_year']:>8.2f}"
        )


if __name__ == "__main__":
    main()
