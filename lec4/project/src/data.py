from __future__ import annotations

import numpy as np

# 生成二维 Sinc 函数数据集
# @param n_total: 生成的数据点总数
# @param domain_min: 输入数据的最小值
# @param domain_max: 输入数据的最大值
# @param min_radius: 输入数据的最小半径
# @param rng: 随机数生成器
# @return: 输入数据和对应的 Sinc 函数值

def sample_sinc_data(
    n_total: int,
    domain_min: float,
    domain_max: float,
    min_radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.empty((2, n_total), dtype=np.float64)
    targets = np.empty((1, n_total), dtype=np.float64)
    generated = 0

    while generated < n_total:
        remain = n_total - generated
        candidates = rng.uniform(domain_min, domain_max, size=(2, remain)) 
        radii = np.linalg.norm(candidates, axis=0)
        mask = radii >= min_radius
        valid = candidates[:, mask]
        count = valid.shape[1]
        if count == 0:
            continue
        inputs[:, generated : generated + count] = valid
        r_valid = radii[mask]
        targets[:, generated : generated + count] = np.sinc(r_valid / np.pi)
        generated += count

    return inputs, targets
