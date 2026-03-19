# from project.solver import Solver
from __future__ import annotations

from dataclasses import dataclass
import timeit
import numpy as np


ArrayLike = np.ndarray


@dataclass
class WeiszfeldResult:
    x_star: ArrayLike
    objective_value: float
    iterations: int
    converged: bool
    history: list[ArrayLike]
    objective_history: list[float]


def euclidean_distance(a: ArrayLike, x: ArrayLike) -> float:
    return float(np.linalg.norm(a - x))


def objective(x: ArrayLike, points: ArrayLike, weights: ArrayLike) -> float:
    distances = np.linalg.norm(points - x, axis=1)
    return float(np.sum(weights * distances))


def gradient(x: ArrayLike, points: ArrayLike, weights: ArrayLike) -> ArrayLike:
    diff = x - points
    distances = np.linalg.norm(diff, axis=1)

    if np.any(distances == 0):
        raise ValueError(
            "Gradient is not defined at an anchor point. "
            "The algorithm should handle this case separately."
        )

    return np.sum(weights[:, None] * diff / distances[:, None], axis=0)


def convex_hull_radius_upper_bound(x: ArrayLike, points: ArrayLike) -> float:
    """
    Computes a valid upper bound for
    sigma(x) = max{ ||x - y||_2 : y in conv(points) }.

    Since the maximum of a convex function over a compact convex polytope
    is attained at an extreme point, it is enough to check the anchor points.
    """
    return float(np.max(np.linalg.norm(points - x, axis=1)))


def test_value_at_anchor(k: int, points: ArrayLike, weights: ArrayLike) -> float:
    """
    Computes Test_k from the project statement.
    """
    ak = points[k]
    mask = np.ones(len(points), dtype=bool)
    mask[k] = False

    other_points = points[mask]
    other_weights = weights[mask]

    diff = ak - other_points
    distances = np.linalg.norm(diff, axis=1)

    term = np.sum(other_weights[:, None] * diff / distances[:, None], axis=0)
    return float(np.linalg.norm(term))


def minimizer_is_anchor(points: ArrayLike, weights: ArrayLike) -> tuple[bool, int | None]:
    """
    Checks Step (a) in the project statement.
    Returns (True, k) if a_k is a global minimizer, otherwise (False, None).
    """
    for k in range(len(points)):
        if test_value_at_anchor(k, points, weights) <= weights[k]:
            return True, k
    return False, None


def squared_problem_minimizer(points: ArrayLike, weights: ArrayLike) -> ArrayLike:
    """
    Minimizer of the weighted squared Euclidean median problem:
    min_x sum_i v_i ||x - a^i||_2^2
    """
    total_weight = np.sum(weights)
    if total_weight <= 0:
        raise ValueError("The sum of weights must be positive.")
    return np.sum(weights[:, None] * points, axis=0) / total_weight


def weiszfeld_update(x: ArrayLike, points: ArrayLike, weights: ArrayLike) -> ArrayLike:
    distances = np.linalg.norm(points - x, axis=1)

    if np.any(distances == 0):
        raise ValueError(
            "Weiszfeld update is not defined when the current iterate equals an anchor point."
        )

    inv_dist = weights / distances
    numerator = np.sum(inv_dist[:, None] * points, axis=0)
    denominator = np.sum(inv_dist)
    return numerator / denominator


def relative_error_bound(x: ArrayLike, points: ArrayLike, weights: ArrayLike) -> float | None:
    """
    Computes the bound from Theorem 4:
    UB(x) / LB(x),
    where
        UB(x) = ||grad f(x)|| * sigma(x),
        LB(x) = f(x) - ||grad f(x)|| * sigma(x).

    Returns None if LB(x) <= 0.
    """
    fx = objective(x, points, weights)
    grad_fx = gradient(x, points, weights)
    sigma_x = convex_hull_radius_upper_bound(x, points)

    ub = float(np.linalg.norm(grad_fx) * sigma_x)
    lb = float(fx - ub)

    if lb <= 0:
        return None

    return ub / lb


def weiszfeld(
    points: ArrayLike,
    weights: ArrayLike | None = None,
    x0: ArrayLike | None = None,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    use_theorem4_stop: bool = True,
) -> WeiszfeldResult:
    """
    Solves
        min_x sum_i v_i ||x - a^i||_2
    using the Weiszfeld algorithm.

    Parameters
    ----------
    points : np.ndarray of shape (m, 2)
        Anchor points a^i.
    weights : np.ndarray of shape (m,), optional
        Nonnegative weights. Defaults to all ones.
    x0 : np.ndarray of shape (2,), optional
        Starting point. If omitted, uses the minimizer of the weighted squared problem.
    tol : float
        Tolerance for stopping.
    max_iter : int
        Maximum number of iterations.
    use_theorem4_stop : bool
        If True, uses the relative bound from Theorem 4 when possible.
        Otherwise falls back to ||x^{k+1} - x^k||_2 < tol.

    Returns
    -------
    WeiszfeldResult
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (m, 2).")

    m = len(points)
    if m == 0:
        raise ValueError("At least one point is required.")

    if weights is None:
        weights = np.ones(m, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if weights.shape != (m,):
        raise ValueError("weights must have shape (m,).")

    if np.any(weights <= 0):
        raise ValueError("All weights must be strictly positive for the algorithm.")

    anchor_solution, anchor_index = minimizer_is_anchor(points, weights)
    if anchor_solution:
        x_star = points[anchor_index].copy()
        return WeiszfeldResult(
            x_star=x_star,
            objective_value=objective(x_star, points, weights),
            iterations=0,
            converged=True,
            history=[x_star.copy()],
            objective_history=[objective(x_star, points, weights)],
        )

    if x0 is None:
        x = squared_problem_minimizer(points, weights)
    else:
        x = np.asarray(x0, dtype=float)
        if x.shape != (2,):
            raise ValueError("x0 must have shape (2,).")

    if np.any(np.linalg.norm(points - x, axis=1) == 0):
        x = x + 1e-6

    history = [x.copy()]
    objective_history = [objective(x, points, weights)]

    for iteration in range(1, max_iter + 1):
        x_new = weiszfeld_update(x, points, weights)
        history.append(x_new.copy())
        objective_history.append(objective(x_new, points, weights))

        if use_theorem4_stop:
            try:
                bound = relative_error_bound(x_new, points, weights)
            except ValueError:
                bound = None

            if bound is not None and bound < tol:
                return WeiszfeldResult(
                    x_star=x_new,
                    objective_value=objective_history[-1],
                    iterations=iteration,
                    converged=True,
                    history=history,
                    objective_history=objective_history,
                )

        if np.linalg.norm(x_new - x) < tol:
            return WeiszfeldResult(
                x_star=x_new,
                objective_value=objective_history[-1],
                iterations=iteration,
                converged=True,
                history=history,
                objective_history=objective_history,
            )

        x = x_new

    return WeiszfeldResult(
        x_star=x,
        objective_value=objective_history[-1],
        iterations=max_iter,
        converged=False,
        history=history,
        objective_history=objective_history,
    )


def print_result(name: str, result: WeiszfeldResult) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"x*: {result.x_star}")
    print(f"f(x*): {result.objective_value:.12f}")


def run_test_case_1() -> None:
    """
    Problem 13(i):
    Unit square with equal weights.
    Expected minimizer: (0, 0)^T.
    """
    points = np.array(
        [
            [-1.0, -1.0],
            [-1.0,  1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
        ]
    )
    weights = np.ones(4)

    result = weiszfeld(points, weights, tol=1e-10, max_iter=10_000)
    print_result("Test case 1: Unit square", result)

    expected = np.array([0.0, 0.0])
    error = np.linalg.norm(result.x_star - expected)
    print(f"Distance to expected minimizer (0,0): {error:.12e}")


def run_test_case_2() -> None:
    """
    Problem 13(ii):
    Less symmetric example with nonuniform weights and
    a manually chosen starting point to make the iteration visible.
    """
    points = np.array(
        [
            [0.0, 0.0],
            [4.0, 1.0],
            [6.0, 4.0],
            [9.0, 0.0],
            [10.0, 5.0],
        ]
    )
    weights = np.array([1.0, 2.0, 1.0, 4.0, 2.0])

    x0 = np.array([-2.0, 6.0])

    result = weiszfeld(
        points,
        weights,
        x0=x0,
        tol=1e-5,
        max_iter=10_000,
    )

    print_result("Test case 2: Asymmetric weighted example", result)

    print("\nIteration history:")
    for k, x in enumerate(result.history[:10]):
        print(f"k = {k:2d}, x = {x}, f(x) = {result.objective_history[k]:.12f}")

    if len(result.history) > 10:
        print("...")

def main() -> None:
    run_test_case_1()
    run_test_case_2()

    runs = 10
    print(f"\nTiming run_test_case_2() with timeit ({runs} runs)...")
    timings = timeit.repeat(
        stmt=lambda: run_test_case_2(),
        repeat=runs,
        number=1,
    )

    for i, t in enumerate(timings, start=1):
        print(f"Run {i:2d}: {t:.6f} s")

    print(f"Best time: {min(timings):.6f} s")
    print(f"Average time: {float(np.mean(timings)):.6f} s")

if __name__ == "__main__":
    main()