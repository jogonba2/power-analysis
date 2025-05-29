import numpy as np
import pytest

from power.metrics import find_dataset_size, find_minimum_detectable_effect


@pytest.mark.parametrize(
    ["mde", "alpha", "beta", "var_a", "var_b", "omega", "k_a", "k_b"],
    [(0.03, 0.05, 0.2, 0.0, 0.0, 1 / 9, 1, 1)],
)
def test_find_dataset_size_from_params(
    mde: float,
    alpha: float,
    beta: float,
    var_a: float,
    var_b: float,
    omega: float,
    k_a: int,
    k_b: int,
):
    dataset_size = find_dataset_size(
        mde,
        alpha=alpha,
        beta=beta,
        var_a=var_a,
        var_b=var_b,
        omega=omega,
        k_a=k_a,
        k_b=k_b,
    )
    assert dataset_size == 968


@pytest.fixture
def x_a():
    return [0.1, 0.2, 0.3]


@pytest.fixture
def x_b():
    return [0.2, 0.4, 0.7]


@pytest.mark.parametrize(["mde", "alpha", "beta"], [(0.03, 0.05, 0.2)])
def test_find_dataset_size_from_data(
    mde: float,
    alpha: float,
    beta: float,
    x_a: list | np.ndarray,
    x_b: list | np.ndarray,
):
    dataset_size = find_dataset_size(
        mde, alpha=alpha, beta=beta, x_a=x_a, x_b=x_b
    )
    assert dataset_size == 132


@pytest.mark.parametrize(
    ["dataset_size", "alpha", "beta", "var_a", "var_b", "omega", "k_a", "k_b"],
    [(968, 0.05, 0.2, 0.0, 0.0, 1 / 9, 1, 1)],
)
def test_find_minimum_detectable_effect_from_params(
    dataset_size: int,
    alpha: float,
    beta: float,
    var_a: float,
    var_b: float,
    omega: float,
    k_a: int,
    k_b: int,
):
    mde = find_minimum_detectable_effect(
        dataset_size,
        alpha=alpha,
        beta=beta,
        var_a=var_a,
        var_b=var_b,
        omega=omega,
        k_a=k_a,
        k_b=k_b,
    )

    np.testing.assert_almost_equal(mde, 0.03, decimal=3)


@pytest.mark.parametrize(
    ["dataset_size", "alpha", "beta"],
    [(132, 0.05, 0.2)],
)
def test_find_minimum_detectable_effect_from_params(
    dataset_size: int,
    alpha: float,
    beta: float,
    x_a: list | np.ndarray,
    x_b: list | np.ndarray,
):
    mde = find_minimum_detectable_effect(
        dataset_size, alpha=alpha, beta=beta, x_a=x_a, x_b=x_b
    )

    np.testing.assert_almost_equal(mde, 0.03, decimal=3)
