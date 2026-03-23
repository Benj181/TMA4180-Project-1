from __future__ import annotations

from dataclasses import dataclass
import timeit
import numpy as np


ArrayLike = np.ndarray


@dataclass
class OptimizationResult:
    x_star: ArrayLike
    objective_value: float
    iterations: int
    converged: bool
    history: list[ArrayLike]
    objective_history: list[float]


def objective(x: ArrayLike, points: ArrayLike, weights: ArrayLike) -> float:
    distances = np.linalg.norm(points - x, axis=1)
    return float(np.sum(weights * distances))


def gradient(x: ArrayLike, points: ArrayLike, weights: ArrayLike) -> ArrayLike:
    diff = x - points
    distances = np.linalg.norm(diff, axis=1)

    if np.any(distances == 0):
        raise ValueError(
            "Gradient is not defined at an anchor point."
        )

    return np.sum(weights[:, None] * diff / distances[:, None], axis=0)


def convex_hull_radius_upper_bound(x: ArrayLike, points: ArrayLike) -> float:
    return float(np.max(np.linalg.norm(points - x, axis=1)))


def relative_error_bound(x: ArrayLike, points: ArrayLike, weights: ArrayLike) -> float | None:
    fx = objective(x, points, weights)
    grad_fx = gradient(x, points, weights)
    sigma_x = convex_hull_radius_upper_bound(x, points)

    ub = float(np.linalg.norm(grad_fx) * sigma_x)
    lb = float(fx - ub)

    if lb <= 0:
        return None

    return ub / lb


def test_value_at_anchor(k: int, points: ArrayLike, weights: ArrayLike) -> float:
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
    for k in range(len(points)):
        if test_value_at_anchor(k, points, weights) <= weights[k]:
            return True, k
    return False, None


def squared_problem_minimizer(points: ArrayLike, weights: ArrayLike) -> ArrayLike:
    total_weight = np.sum(weights)
    if total_weight <= 0:
        raise ValueError("The sum of weights must be positive.")
    return np.sum(weights[:, None] * points, axis=0) / total_weight


def backtracking_line_search(
    x: ArrayLike,
    grad: ArrayLike,
    points: ArrayLike,
    weights: ArrayLike,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    min_alpha: float = 1e-14,
) -> float:
    """
    Armijo backtracking line search.

    Finds alpha such that
    f(x - alpha * grad) <= f(x) - c * alpha * ||grad||^2.
    """
    fx = objective(x, points, weights)
    grad_norm_sq = float(np.dot(grad, grad))

    alpha = alpha0
    while alpha >= min_alpha:
        x_new = x - alpha * grad
        if objective(x_new, points, weights) <= fx - c * alpha * grad_norm_sq:
            return alpha
        alpha *= rho

    return min_alpha


def gradient_descent_backtracking(
    points: ArrayLike,
    weights: ArrayLike | None = None,
    x0: ArrayLike | None = None,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    use_theorem4_stop: bool = True,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
) -> OptimizationResult:
    """
    Solves
        min_x sum_i v_i ||x - a^i||_2
    using gradient descent with backtracking.

    This replaces steps (b)-(d) of the Weiszfeld algorithm.
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
        raise ValueError("All weights must be strictly positive.")

    # Step (a): same anchor test as in the Weiszfeld algorithm
    anchor_solution, anchor_index = minimizer_is_anchor(points, weights)
    if anchor_solution:
        x_star = points[anchor_index].copy()
        fx = objective(x_star, points, weights)
        return OptimizationResult(
            x_star=x_star,
            objective_value=fx,
            iterations=0,
            converged=True,
            history=[x_star.copy()],
            objective_history=[fx],
        )

    # Starting point
    if x0 is None:
        x = squared_problem_minimizer(points, weights)
    else:
        x = np.asarray(x0, dtype=float)
        if x.shape != (2,):
            raise ValueError("x0 must have shape (2,).")

    # Avoid starting exactly at an anchor point
    if np.any(np.linalg.norm(points - x, axis=1) == 0):
        x = x + 1e-6

    history = [x.copy()]
    objective_history = [objective(x, points, weights)]

    for iteration in range(1, max_iter + 1):
        try:
            grad = gradient(x, points, weights)
        except ValueError:
            # If x lands exactly on an anchor point, stop safely
            fx = objective(x, points, weights)
            return OptimizationResult(
                x_star=x,
                objective_value=fx,
                iterations=iteration - 1,
                converged=True,
                history=history,
                objective_history=objective_history,
            )

        grad_norm = float(np.linalg.norm(grad))

        # Basic stationary-point stop
        if grad_norm < tol:
            fx = objective(x, points, weights)
            return OptimizationResult(
                x_star=x,
                objective_value=fx,
                iterations=iteration - 1,
                converged=True,
                history=history,
                objective_history=objective_history,
            )

        alpha = backtracking_line_search(
            x=x,
            grad=grad,
            points=points,
            weights=weights,
            alpha0=alpha0,
            rho=rho,
            c=c,
        )

        x_new = x - alpha * grad
        history.append(x_new.copy())
        objective_history.append(objective(x_new, points, weights))

        if use_theorem4_stop:
            try:
                bound = relative_error_bound(x_new, points, weights)
            except ValueError:
                bound = None

            if bound is not None and bound < tol:
                return OptimizationResult(
                    x_star=x_new,
                    objective_value=objective_history[-1],
                    iterations=iteration,
                    converged=True,
                    history=history,
                    objective_history=objective_history,
                )

        if np.linalg.norm(x_new - x) < tol:
            return OptimizationResult(
                x_star=x_new,
                objective_value=objective_history[-1],
                iterations=iteration,
                converged=True,
                history=history,
                objective_history=objective_history,
            )

        x = x_new

    return OptimizationResult(
        x_star=x,
        objective_value=objective_history[-1],
        iterations=max_iter,
        converged=False,
        history=history,
        objective_history=objective_history,
    )


def print_result(name: str, result: OptimizationResult) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"x*: {result.x_star}")
    print(f"f(x*): {result.objective_value:.12f}")


def run_gd_test_case_1() -> None:
    """
    Problem 13(i):
    Unit square with equal weights.
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

    result = gradient_descent_backtracking(
        points=points,
        weights=weights,
        x0=np.array([0.3, -0.8]),   # choose non-optimal start to see iterations
        tol=1e-10,
        max_iter=10_000,
        use_theorem4_stop=True,
        alpha0=1.0,
        rho=0.5,
        c=1e-4,
    )
    print_result("Gradient descent test case 1: Unit square", result)


def run_gd_test_case_2() -> None:
    """
    Asymmetric weighted example.
    """
    points = np.array([[-4,0],[-2,1],[-2,-1],[2,0],[8,0]])
    
    weights = np.array([1.0, 1.0, 1.0, 2.0, 3.0])

    result = gradient_descent_backtracking(
        points=points,
        weights=weights,
        x0=np.array([-2.0, 6.0]),
        tol=1e-9,
        max_iter=10_000,
        use_theorem4_stop=True,
        alpha0=1.0,
        rho=0.5,
        c=1e-4,
    )
    print_result("Gradient descent test case 2: Asymmetric weighted example", result)

    print("\nFirst iterations:")
    for k, x in enumerate(result.history[:10]):
        print(f"k = {k:2d}, x = {x}, f(x) = {result.objective_history[k]:.12f}")
    if len(result.history) > 10:
        print("...")


def main() -> None:
    run_gd_test_case_1()

    run_gd_test_case_2()

    runs = 10
    print(f"\nTiming run_gd_test_case_2() with timeit ({runs} runs)...")
    timings = timeit.repeat(
        stmt=lambda: run_gd_test_case_2(),
        repeat=runs,
        number=1,
    )

    for i, t in enumerate(timings, start=1):
        print(f"Run {i:2d}: {t:.6f} s")

    print(f"Best time: {min(timings):.6f} s")
    print(f"Average time: {float(np.mean(timings)):.6f} s")


if __name__ == "__main__":
    main()