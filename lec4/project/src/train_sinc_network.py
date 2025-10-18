from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from data import sample_sinc_data
from model import MLP, initialize_mlp


@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_loss: List[float]

# 计算均方误差
def _compute_mse(model: MLP, inputs: np.ndarray, targets: np.ndarray) -> float:
    preds = model.predict(inputs)
    return float(np.mean((preds - targets) ** 2))

# 计算梯度
def _compute_gradients(model: MLP, activations: List[np.ndarray], targets: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
    batch_size = targets.shape[1]
    delta = 2.0 * (activations[-1] - targets) / batch_size
    grad_w = [np.zeros_like(w) for w in model.weights]
    grad_b = [np.zeros_like(b) for b in model.biases]

    for idx in reversed(range(len(model.weights))):
        grad_w[idx] = delta @ activations[idx].T
        grad_b[idx] = np.sum(delta, axis=1, keepdims=True)
        if idx > 0:
            delta = (model.weights[idx].T @ delta) * (1.0 - activations[idx] ** 2)

    return grad_w, grad_b

# 应用梯度更新模型参数
def _apply_gradients(
    model: MLP,
    grad_w: List[np.ndarray],
    grad_b: List[np.ndarray],
    lr: float,
    optimizer_state: Dict | None,
) -> Dict | None:
    if optimizer_state is None or optimizer_state.get("type", "sgd") == "sgd":
        for idx in range(len(model.weights)):
            model.weights[idx] -= lr * grad_w[idx]
            model.biases[idx] -= lr * grad_b[idx]
        return optimizer_state

    if optimizer_state.get("type") != "adam":
        raise ValueError(f"Unsupported optimizer: {optimizer_state.get('type')}")

    beta1 = optimizer_state["beta1"]
    beta2 = optimizer_state["beta2"]
    eps = optimizer_state["eps"]
    optimizer_state["t"] += 1
    t = optimizer_state["t"]

    for idx in range(len(model.weights)):

        optimizer_state["m_w"][idx] = beta1 * optimizer_state["m_w"][idx] + (1.0 - beta1) * grad_w[idx]
        optimizer_state["v_w"][idx] = beta2 * optimizer_state["v_w"][idx] + (1.0 - beta2) * (grad_w[idx] ** 2)

        m_w_hat = optimizer_state["m_w"][idx] / (1.0 - beta1 ** t)
        v_w_hat = optimizer_state["v_w"][idx] / (1.0 - beta2 ** t)

        model.weights[idx] -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

        optimizer_state["m_b"][idx] = beta1 * optimizer_state["m_b"][idx] + (1.0 - beta1) * grad_b[idx]
        optimizer_state["v_b"][idx] = beta2 * optimizer_state["v_b"][idx] + (1.0 - beta2) * (grad_b[idx] ** 2)

        m_b_hat = optimizer_state["m_b"][idx] / (1.0 - beta1 ** t)
        v_b_hat = optimizer_state["v_b"][idx] / (1.0 - beta2 ** t)

        model.biases[idx] -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    return optimizer_state

# 训练 Sinc 网络
# @param config: 配置字典，包含训练参数
# @return: 训练结果字典，包含模型、历史记录、测试误差
def train_sinc_network(config: Dict) -> Dict:
    rng = np.random.default_rng(config.get("rng_seed"))

    inputs, targets = sample_sinc_data(
        config["N_total"],
        config["domain_min"],
        config["domain_max"],
        config["min_radius"],
        rng,
    )

    n_total = inputs.shape[1]
    n_train = int(round(0.7 * n_total))
    n_val = int(round(0.15 * n_total))

    permutation = rng.permutation(n_total)
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train : n_train + n_val]
    test_idx = permutation[n_train + n_val :]

    train_inputs = inputs[:, train_idx]
    train_targets = targets[:, train_idx]
    val_inputs = inputs[:, val_idx]
    val_targets = targets[:, val_idx]
    test_inputs = inputs[:, test_idx]
    test_targets = targets[:, test_idx]

    layer_sizes = [inputs.shape[0], *config["hidden_sizes"], 1]
    model = initialize_mlp(layer_sizes, rng)

    input_mean = np.mean(train_inputs, axis=1, keepdims=True)
    input_std = np.std(train_inputs, axis=1, keepdims=True)
    input_std = np.where(input_std < 1e-6, 1.0, input_std)
    model.set_input_normalization(input_mean, input_std)

    max_epochs = config["max_epochs"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    optimizer_type = config.get("optimizer", "adam").lower()

    if optimizer_type == "adam":
        optimizer_state: Dict | None = {
            "type": "adam",
            "beta1": config.get("beta1", 0.9),
            "beta2": config.get("beta2", 0.999),
            "eps": config.get("epsilon", 1e-8),
            "m_w": [np.zeros_like(w) for w in model.weights],
            "v_w": [np.zeros_like(w) for w in model.weights],
            "m_b": [np.zeros_like(b) for b in model.biases],
            "v_b": [np.zeros_like(b) for b in model.biases],
            "t": 0,
        }
    elif optimizer_type == "sgd":
        optimizer_state = {"type": "sgd"}
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    history = TrainingHistory(train_loss=[], val_loss=[])

    for _ in range(max_epochs):
        order = rng.permutation(n_train)
        train_inputs = train_inputs[:, order]
        train_targets = train_targets[:, order]

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_inputs = train_inputs[:, start:end]
            batch_targets = train_targets[:, start:end]
            _, activations = model.forward(batch_inputs)
            grad_w, grad_b = _compute_gradients(model, activations, batch_targets)
            optimizer_state = _apply_gradients(model, grad_w, grad_b, lr, optimizer_state)

        history.train_loss.append(_compute_mse(model, train_inputs, train_targets))
        history.val_loss.append(_compute_mse(model, val_inputs, val_targets))

    pred_test = model.predict(test_inputs)
    abs_denominator = np.maximum(np.abs(test_targets), 1e-8)
    relative_errors = np.abs(pred_test - test_targets) / abs_denominator
    test_error = float(np.mean(relative_errors))

    return {
        "model": model,
        "history": history,
        "test_error": test_error,
        "counts": {
            "total": int(n_total),
            "train": int(n_train),
            "val": int(n_val),
            "test": int(test_idx.size),
        },
        "config": dict(config),
        "input_mean": input_mean,
        "input_std": input_std,
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "val_inputs": val_inputs,
        "val_targets": val_targets,
        "test_inputs": test_inputs,
        "test_targets": test_targets,
    }
