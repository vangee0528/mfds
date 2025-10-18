from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from train_sinc_network import TrainingHistory

preferred_fonts = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.family"] = preferred_fonts
plt.rcParams["font.sans-serif"] = preferred_fonts
plt.rcParams["axes.unicode_minus"] = False


def visualize_sinc_results(results: Dict, config: Dict) -> Dict[str, str]:
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    domain_min = config["domain_min"]
    domain_max = config["domain_max"]
    step = config["grid_step"]

    x_grid, y_grid = np.meshgrid(
        np.arange(domain_min, domain_max + step, step),
        np.arange(domain_min, domain_max + step, step),
    )
    r_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)
    true_grid = np.sinc(r_grid / np.pi)

    grid_inputs = np.vstack((x_grid.ravel(), y_grid.ravel()))
    pred_grid = results["model"].predict(grid_inputs).reshape(x_grid.shape)
    abs_err_grid = np.abs(pred_grid - true_grid)

    saved_paths: Dict[str, str] = {}

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x_grid, y_grid, true_grid, cmap=cm.viridis)
    ax.set_title("二维 Sinc 函数真值")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(surf, shrink=0.6)
    true_path = os.path.join(output_dir, "sinc_true_surface.png")
    fig.savefig(true_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["trueSurface"] = true_path

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x_grid, y_grid, pred_grid, cmap=cm.viridis)
    ax.set_title("神经网络预测的二维 Sinc 函数")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.colorbar(surf, shrink=0.6)
    pred_path = os.path.join(output_dir, "sinc_pred_surface.png")
    fig.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["predSurface"] = pred_path

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x_grid, y_grid, abs_err_grid, cmap=cm.magma)
    ax.set_title("预测绝对误差分布")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("|误差|")
    fig.colorbar(surf, shrink=0.6)
    err_path = os.path.join(output_dir, "sinc_abs_error_surface.png")
    fig.savefig(err_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["errorSurface"] = err_path

    history: TrainingHistory = results["history"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = np.arange(1, len(history.train_loss) + 1)
    ax.semilogy(epochs, history.train_loss, label="训练损失")
    ax.semilogy(epochs, history.val_loss, label="验证损失", linestyle="--")
    ax.set_xlabel("迭代轮数")
    ax.set_ylabel("均方误差")
    ax.set_title("训练/验证性能曲线")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    perf_path = os.path.join(output_dir, "training_performance.png")
    fig.savefig(perf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["trainingPerformance"] = perf_path

    metrics_path = os.path.join(output_dir, "sinc_nn_results.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("自实现前馈网络二维 sinc 拟合实验\n")
        f.write("-----------------------------------\n")
        f.write(f"样本总数: {results['counts']['total']}\n")
        f.write(
            f"训练/验证/测试: {results['counts']['train']} / {results['counts']['val']} / {results['counts']['test']}\n"
        )
        hidden_str = ", ".join(str(v) for v in config["hidden_sizes"])
        f.write(f"网络结构: [{hidden_str}]\n")
        f.write(
            f"训练轮数: {config['max_epochs']}\n学习率: {config['learning_rate']:.4f}, 批大小: {config['batch_size']}\n"
        )
        optimizer_name = config.get("optimizer", "sgd")
        f.write(f"优化器: {optimizer_name}\n")
        if optimizer_name.lower() == "adam":
            f.write(
                f"Adam 参数 β1={config.get('beta1', 0.9):.3f}, β2={config.get('beta2', 0.999):.3f}, ε={config.get('epsilon', 1e-8):.1e}\n"
            )
        f.write(f"最终训练 MSE: {history.train_loss[-1]:.6e}\n")
        f.write(f"最终验证 MSE: {history.val_loss[-1]:.6e}\n")
        f.write(f"测试集平均相对误差: {results['test_error']:.6e}\n")
    saved_paths["metrics"] = metrics_path

    return saved_paths
