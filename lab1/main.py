import numpy as np
from itertools import product
from scipy.optimize import minimize_scalar


class IntervalMatrix:
    """Интервальная матрица: mid ± delta * pattern."""

    def __init__(self, mid, pattern):
        self.mid = np.array(mid, dtype=float)
        self.pattern = np.array(pattern, dtype=float)

    def rad(self, delta):
        return delta * self.pattern

    def contains(self, M, delta, eps=1e-9):
        """Проверка M ∈ A: |M - mid| <= rad (с малым допуском на округление)."""
        return bool(np.all(np.abs(M - self.mid) <= self.rad(delta) + eps))


def rect_min_delta(A):
    """
    Матрица (3x2) имеет неполный столбцовый ранг <=> столбец1 = lam * столбец2.
    Для фиксированного lam условие по каждой строке i:
        |lam - m1_i / m2_i| ... в общем виде: |m1_i - lam*m2_i| <= delta*(r1_i + |lam|*r2_i)
    Отсюда delta(lam) = max_i |m1_i - lam*m2_i| / (r1_i + |lam|*r2_i).
    Минимизируем по lam.
    """
    m1, m2 = A.mid[:, 0], A.mid[:, 1]
    r1, r2 = A.pattern[:, 0], A.pattern[:, 1]

    def f(lam):
        return np.max(np.abs(m1 - lam * m2) / (r1 + abs(lam) * r2))

    res = minimize_scalar(f, bounds=(0.1, 3.0), method="bounded",
                          options={"xatol": 1e-13})
    return res.fun, res.x


def rect_build_singular(A, delta, lam):
    """Строим A' ∈ A со столбцами col1 = lam * col2."""
    m1, m2 = A.mid[:, 0], A.mid[:, 1]
    r1, r2 = A.rad(delta)[:, 0], A.rad(delta)[:, 1]
    # b_i должно лежать и в [m2±r2], и в [(m1±r1)/lam]
    lo = np.maximum(m2 - r2, (m1 - r1) / lam)
    hi = np.minimum(m2 + r2, (m1 + r1) / lam)
    b = (lo + hi) / 2
    return np.column_stack([lam * b, b])


def solve_rect(name, A):
    delta, lam = rect_min_delta(A)
    M = rect_build_singular(A, delta, lam)
    sigma_min = np.linalg.svd(M, compute_uv=False)[-1]
    print(f"\n[{name}]")
    print(f"  δ_min = {delta:.6f}  (особенная при δ >= δ_min), lam = {lam:.6f}")
    print("  A' =\n", M)
    print("  A' ∈ A:", A.contains(M, delta), "| σ_min(A') =", f"{sigma_min:.2e}")

def square_min_delta(A):
 
    n = A.mid.shape[0]
    C_inv = np.linalg.inv(A.mid)
    best = (-1, None)
    for y in product([-1, 1], repeat=n):
        y = np.array(y)
        s = np.sum(np.abs(C_inv @ y))
        if s > best[0]:
            best = (s, y)
    s, y = best
    z = np.sign(C_inv @ y)
    z[z == 0] = 1  
    return 1 / s, y, z


def square_build_singular(A, delta, y, z):
    """A' = C - delta * y z^T  (лежит на границе A, det A' = 0)."""
    return A.mid - delta * np.outer(y, z)


def solve_square(A):
    delta, y, z = square_min_delta(A)
    M = square_build_singular(A, delta, y, z)
    print("\n[Квадратная A2]")
    print(f"  δ_min = {delta:.6f}  (особенная при δ >= δ_min)")
    print("  A' =\n", M)
    print("  A' ∈ A:", A.contains(M, delta), "| det A' =", f"{np.linalg.det(M):.2e}")


def i_mul(a, b):
    p = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return min(p), max(p)


def i_add(a, b):
    return a[0] + b[0], a[1] + b[1]


def i_neg(a):
    return -a[1], -a[0]


def naive_det3(A, delta):
    """Естественное интервальное расширение формулы Саррюса для 3x3."""
    low, high = A.mid - A.rad(delta), A.mid + A.rad(delta)
    el = lambda i, j: (low[i, j], high[i, j])
    triple = lambda i, j, k, l, m, n: i_mul(i_mul(el(i, j), el(k, l)), el(m, n))

    plus = [triple(0, 0, 1, 1, 2, 2), triple(0, 1, 1, 2, 2, 0), triple(0, 2, 1, 0, 2, 1)]
    minus = [triple(0, 2, 1, 1, 2, 0), triple(0, 0, 1, 2, 2, 1), triple(0, 1, 1, 0, 2, 2)]

    res = (0.0, 0.0)
    for t in plus:
        res = i_add(res, t)
    for t in minus:
        res = i_add(res, i_neg(t))
    return res


def naive_min_delta(A, upper=0.2):
    """Бисекция: наименьшее delta, при котором 0 ∈ det(A) (наивная оценка)."""
    left, right = 0.0, upper
    for _ in range(60):
        m = (left + right) / 2
        if naive_det3(A, m)[0] <= 0:
            right = m
        else:
            left = m
    return right


if __name__ == "__main__":
    mid1 = [[0.95, 1.0], [1.05, 1.0], [1.10, 1.0]]
    tomography = [[1, 1], [1, 1], [1, 1]]
    regression = [[1, 0], [1, 0], [1, 0]]

    solve_rect("A1, томография", IntervalMatrix(mid1, tomography))
    solve_rect("A1, регрессия", IntervalMatrix(mid1, regression))

    mid2 = [[1.1, 0.9, 1.1], [1.4, 1.0, 0.8], [0.8, 1.4, 1.2]]
    A2 = IntervalMatrix(mid2, np.ones((3, 3)))
    solve_square(A2)

    true_delta = square_min_delta(A2)[0]
    naive = naive_min_delta(A2)
    print(f"\n[Наивный интервальный определитель] δ_наив = {naive:.6f}, "
          f"δ_истинн / δ_наив = {true_delta / naive:.2f}")