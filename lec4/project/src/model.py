from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# 多层感知机 类
@dataclass
class MLP:
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None

    def set_input_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.input_mean = mean
        self.input_std = std

    def _normalize_inputs(self, inputs: np.ndarray) -> np.ndarray:
        if self.input_mean is None or self.input_std is None:
            return inputs
        return (inputs - self.input_mean) / self.input_std

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        normalized_inputs = self._normalize_inputs(inputs)
        activations: List[np.ndarray] = [normalized_inputs]
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ activations[-1] + b
            if idx == len(self.weights) - 1:
                activations.append(z)
            else:
                activations.append(np.tanh(z))
        return activations[-1], activations

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        output, _ = self.forward(inputs)
        return output


def initialize_mlp(layer_sizes: List[int], rng: np.random.Generator) -> MLP:
    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(rng.standard_normal(size=(fan_out, fan_in)))
        biases.append(rng.standard_normal(size=(fan_out, 1)))
    return MLP(weights=weights, biases=biases)
