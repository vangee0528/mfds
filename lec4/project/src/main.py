from __future__ import annotations

from pathlib import Path

from train_sinc_network import train_sinc_network
from visualize import visualize_sinc_results

# 主函数
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "output" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "rng_seed": 2025,
        "N_total": 8000,
        "domain_min": -8.0,
        "domain_max": 8.0,
        "min_radius": 1e-6,
    "hidden_sizes": [20, 20],
    "max_epochs": 1000,
    "learning_rate": 0.001,
    "batch_size": 64,
        "optimizer": "adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "grid_step": 0.25,
        "output_dir": str(output_dir),
    }

    results = train_sinc_network(config)
    print(f"测试集平均相对误差: {results['test_error']:.6f}")

    saved_paths = visualize_sinc_results(results, config)
    for key, path in saved_paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
