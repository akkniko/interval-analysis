"""Лабораторная работа № 3 по интервальному анализу. Вариант 5.

Реализация основана на распознающем функционале Tol из учебника
А.Н. Баженова, А.А. Карповой «Интервальный анализ в примерах», §1.4.6,
§1.6.1–1.6.4.

Зависимости:
    pip install numpy scipy matplotlib

Результат:
    - в консоли: максимумы Tol, argmax, активные линейные образующие,
      результаты коррекций и сравнение искажений;
    - в папке lab3_variant5_plots: графики функционала и допусковых множеств.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
import sys, io
sys.stdout.reconfigure(encoding='utf-8')


# Небольшой положительный запас Tol после коррекции
# При margin=0 получается граничный случай max(Tol)=0.
CORRECTION_MARGIN = 0.005

# Для Ab-коррекции половина необходимого прироста Tol достигается уширением b,
# оставшаяся часть — сужением интервалов матрицы A.
AB_B_SHARE = 0.5

GRID_SIZE = 101
FEASIBLE_GRID_SIZE = 241
SAVE_PLOTS = True
SHOW_PLOTS = False
PLOTS_DIR = Path(__file__).resolve().parent / "lab3_variant5_plots"


# A1: 2x2; A2: 3x2; A3: 4x2.
A1_LO = np.array([
    [0.78, 0.79],
    [0.86, 0.83],
], dtype=float)
A1_HI = np.array([
    [1.12, 1.21],
    [1.24, 1.23],
], dtype=float)

A2_LO = np.vstack([
    A1_LO,
    [1.00, 0.76],
])
A2_HI = np.vstack([
    A1_HI,
    [1.20, 1.16],
])

A3_LO = np.vstack([
    A2_LO,
    [-0.13, 0.87],
])
A3_HI = np.vstack([
    A2_HI,
    [0.13, 1.25],
])

B1_LO = np.array([5.00, 5.05], dtype=float)
B1_HI = np.array([5.05, 5.10], dtype=float)

B2_LO = np.array([5.00, 5.05, 5.07], dtype=float)
B2_HI = np.array([5.05, 5.10, 5.12], dtype=float)

B3_LO = np.array([5.00, 5.05, 5.07, 4.00], dtype=float)
B3_HI = np.array([5.05, 5.10, 5.12, 4.05], dtype=float)


@dataclass
class IntervalSystem:
    name: str
    a_lo: np.ndarray
    a_hi: np.ndarray
    b_lo: np.ndarray
    b_hi: np.ndarray

    def copy(self, name: str | None = None) -> "IntervalSystem":
        return IntervalSystem(
            name=name or self.name,
            a_lo=self.a_lo.copy(),
            a_hi=self.a_hi.copy(),
            b_lo=self.b_lo.copy(),
            b_hi=self.b_hi.copy(),
        )


SYSTEMS = [
    IntervalSystem("A1", A1_LO, A1_HI, B1_LO, B1_HI),
    IntervalSystem("A2", A2_LO, A2_HI, B2_LO, B2_HI),
    IntervalSystem("A3", A3_LO, A3_HI, B3_LO, B3_HI),
]


# Интервальные характеристики и распознающий функционал Tol.
# Tol(x,A,b) = min_i { rad(b_i) - |mid(b_i) - sum_j A_ij*x_j| }.
# Произведения A_ij*x_j и сумма по строке вычисляются как интервальные.

def interval_mid_rad(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Середины и радиусы интервалов."""
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def ax_interval_bounds(
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Нижняя и верхняя границы интервального вектора A*x для точечного x."""
    products_1 = a_lo * x[np.newaxis, :]
    products_2 = a_hi * x[np.newaxis, :]
    lower = np.minimum(products_1, products_2).sum(axis=1)
    upper = np.maximum(products_1, products_2).sum(axis=1)
    return lower, upper


def row_tolerances(system: IntervalSystem, x: np.ndarray) -> np.ndarray:
    """Значения слагаемых по строкам, минимум которых равен Tol(x,A,b)."""
    mid_b, rad_b = interval_mid_rad(system.b_lo, system.b_hi)
    lower_ax, upper_ax = ax_interval_bounds(system.a_lo, system.a_hi, x)
    residual_magnitude = np.maximum(
        np.abs(mid_b - lower_ax),
        np.abs(mid_b - upper_ax),
    )
    return rad_b - residual_magnitude


def tol_value(system: IntervalSystem, x: np.ndarray) -> float:
    """Распознающий функционал Tol в точке x."""
    return float(np.min(row_tolerances(system, np.asarray(x, dtype=float))))


# Максимизация Tol.
# В каждой из 2^n областей знаков координат x интервальные концы A*x являются
# линейными функциями. Абсолютные значения раскрываются двумя неравенствами.
# Поэтому максимум нижней огибающей можно найти линейным программированием.


@dataclass
class MaximumResult:
    value: float
    x: np.ndarray
    signs: tuple[int, ...]


def maximize_tol(system: IntervalSystem) -> MaximumResult:
    """Находит глобальный максимум Tol перебором квадрантов и решением LP."""
    a_lo, a_hi = system.a_lo, system.a_hi
    mid_b, rad_b = interval_mid_rad(system.b_lo, system.b_hi)
    m, n = a_lo.shape

    best: MaximumResult | None = None

    for signs in product((1, -1), repeat=n):
        signs_array = np.asarray(signs)

        # В выбранной области знаков это коэффициенты нижней/верхней
        # границы интервального произведения A*x.
        lower_coeff = np.where(signs_array[np.newaxis, :] > 0, a_lo, a_hi)
        upper_coeff = np.where(signs_array[np.newaxis, :] > 0, a_hi, a_lo)

        # Переменные LP: x_1, ..., x_n, t. Максимизируем t.
        # Для каждого конца E(x) интервального A_i*x:
        # t <= rad(b_i) - |mid(b_i) - E(x)|.
        # Это эквивалентно двум линейным ограничениям:
        # t - E(x) <= rad(b_i) - mid(b_i),
        # t + E(x) <= rad(b_i) + mid(b_i).
        a_ub: list[np.ndarray] = []
        b_ub: list[float] = []

        for i in range(m):
            for endpoint_coeff in (lower_coeff[i], upper_coeff[i]):
                a_ub.append(np.r_[-endpoint_coeff, 1.0])
                b_ub.append(float(rad_b[i] - mid_b[i]))

                a_ub.append(np.r_[endpoint_coeff, 1.0])
                b_ub.append(float(rad_b[i] + mid_b[i]))

        bounds = [
            (0.0, None) if sign > 0 else (None, 0.0)
            for sign in signs
        ] + [(None, None)]

        objective = np.r_[np.zeros(n), -1.0]
        result = linprog(
            objective,
            A_ub=np.asarray(a_ub),
            b_ub=np.asarray(b_ub),
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            if result.status == 3:  # unbounded
                raise RuntimeError(
                    f"Функционал Tol оказался неограниченным в области знаков {signs}."
                )
            continue

        candidate = MaximumResult(
            value=float(result.x[-1]),
            x=np.asarray(result.x[:n], dtype=float),
            signs=tuple(int(s) for s in signs),
        )

        if best is None or candidate.value > best.value:
            best = candidate

    if best is None:
        raise RuntimeError(f"Не удалось максимизировать Tol для системы {system.name}.")

    # Проверка: значение LP должно совпасть с непосредственным вычислением Tol.
    direct_value = tol_value(system, best.x)
    if not np.isclose(best.value, direct_value, atol=1e-7):
        raise RuntimeError(
            "Расхождение между LP и прямым вычислением Tol: "
            f"LP={best.value:.10f}, direct={direct_value:.10f}."
        )

    return best


# Линейные образующие функционала в области знаков argmax.
# В каждой строке получаются четыре линейные функции: два конца A_i*x,
# для каждого из которых раскрывается модуль двумя знаками.


def affine_generators_at_sign_region(
    system: IntervalSystem,
    x: np.ndarray,
) -> list[dict[str, object]]:
    mid_b, rad_b = interval_mid_rad(system.b_lo, system.b_hi)
    signs = np.where(x >= 0.0, 1, -1)
    lower_coeff = np.where(signs[np.newaxis, :] > 0, system.a_lo, system.a_hi)
    upper_coeff = np.where(signs[np.newaxis, :] > 0, system.a_hi, system.a_lo)

    generators: list[dict[str, object]] = []
    for i in range(system.a_lo.shape[0]):
        for endpoint_name, coeff in (
            ("нижний конец A*x", lower_coeff[i]),
            ("верхний конец A*x", upper_coeff[i]),
        ):
            # rad - |mid - E| = min(rad-mid+E, rad+mid-E)
            variants = (
                ("+", coeff.copy(), float(rad_b[i] - mid_b[i])),
                ("-", -coeff.copy(), float(rad_b[i] + mid_b[i])),
            )
            for abs_sign, linear_coeff, constant in variants:
                value = float(constant + linear_coeff @ x)
                generators.append({
                    "label": f"строка {i + 1}, {endpoint_name}, знак {abs_sign}",
                    "coeff": linear_coeff,
                    "constant": constant,
                    "value": value,
                })
    return generators


def print_analysis(system: IntervalSystem, make_plot: bool = True) -> MaximumResult:
    result = maximize_tol(system)
    rows = row_tolerances(system, result.x)
    generators = affine_generators_at_sign_region(system, result.x)
    active = [g for g in generators if abs(float(g["value"]) - result.value) <= 1e-6]

    print("\n" + "=" * 76)
    print(f"СИСТЕМА: {system.name}")
    print("=" * 76)
    print(f"max Tol = {result.value:.9f}")
    print(f"argmax Tol = ({result.x[0]:.9f}, {result.x[1]:.9f})")
    print("Значения строковых функций в точке максимума:")
    for i, value in enumerate(rows, start=1):
        print(f"  g_{i}({result.x[0]:.5f}, {result.x[1]:.5f}) = {value:.9f}")

    print("Активные линейные образующие нижней огибающей:")
    if not active:
        nearest = sorted(generators, key=lambda g: abs(float(g["value"]) - result.value))[:3]
        active = nearest
    for generator in active:
        coeff = np.asarray(generator["coeff"], dtype=float)
        constant = float(generator["constant"])
        print(
            f"  {generator['label']}: "
            f"f(x) = {constant:+.9f} {coeff[0]:+.9f}*x1 {coeff[1]:+.9f}*x2"
        )

    if result.value >= -1e-8:
        print("Допусковое множество: НЕПУСТО (max Tol >= 0).")
    else:
        print("Допусковое множество: ПУСТО (max Tol < 0).")

    if make_plot and SAVE_PLOTS:
        filename = f"{system.name.replace(' ', '_')}.png"
        plot_system(system, result, PLOTS_DIR / filename)

    return result


# b-коррекция: равномерное уширение всех компонент правой части на K.
# При этом max Tol увеличивается ровно на K, argmax не меняется.


def correct_b_uniform(
    system: IntervalSystem,
    k: float,
    name: str,
) -> IntervalSystem:
    corrected = system.copy(name=name)
    corrected.b_lo = system.b_lo - k
    corrected.b_hi = system.b_hi + k
    return corrected


# A-коррекция.
# Сначала пытаемся применить сужение A с сохранением mid(A), как в §1.6.4.
# Если даже точечная матрица mid(A) несовместна, такое симметричное сужение
# не может решить задачу. Тогда используем обобщённую коррекцию: выбираем
# подынтервалы A'_ij внутри исходных A_ij, допускающие смещение середин.

def symmetric_shrink_matrix(
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    remaining_radius_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    mid_a, rad_a = interval_mid_rad(a_lo, a_hi)
    new_rad = remaining_radius_fraction * rad_a
    return mid_a - new_rad, mid_a + new_rad


def find_symmetric_a_correction(
    system: IntervalSystem,
    target_margin: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Ищет максимально широкую симметрично суженную A с Tol_max>=margin.

    Возвращает (A_lo_new, A_hi_new, alpha, achieved_Tol).
    alpha — доля исходного радиуса, которая остаётся после сужения.
    None означает, что даже точечная mid(A) не обеспечивает разрешимость.
    """
    current_max = maximize_tol(system).value
    if current_max >= target_margin:
        return system.a_lo.copy(), system.a_hi.copy(), 1.0, current_max

    mid_a, _ = interval_mid_rad(system.a_lo, system.a_hi)
    point_a_system = system.copy(name=f"{system.name}_point_A")
    point_a_system.a_lo = mid_a.copy()
    point_a_system.a_hi = mid_a.copy()
    point_max = maximize_tol(point_a_system).value

    if point_max < target_margin:
        return None

    # Tol не убывает при уменьшении радиусов A. Ищем наибольшую alpha,
    # при которой max Tol всё ещё не меньше заданного положительного запаса.
    low_alpha, high_alpha = 0.0, 1.0
    for _ in range(45):
        alpha = (low_alpha + high_alpha) / 2.0
        a_lo_new, a_hi_new = symmetric_shrink_matrix(
            system.a_lo, system.a_hi, alpha
        )
        candidate = system.copy(name=f"{system.name}_A_candidate")
        candidate.a_lo = a_lo_new
        candidate.a_hi = a_hi_new
        candidate_max = maximize_tol(candidate).value

        if candidate_max >= target_margin:
            low_alpha = alpha
        else:
            high_alpha = alpha

    a_lo_new, a_hi_new = symmetric_shrink_matrix(
        system.a_lo, system.a_hi, low_alpha
    )
    candidate = system.copy(name=f"{system.name}_A_candidate")
    candidate.a_lo = a_lo_new
    candidate.a_hi = a_hi_new
    achieved = maximize_tol(candidate).value
    return a_lo_new, a_hi_new, low_alpha, achieved


def closest_row_center_inside_original(
    a_lo_row: np.ndarray,
    a_hi_row: np.ndarray,
    target: float,
    x: np.ndarray,
) -> np.ndarray:
    """Ближайший к старой середине центр c при c*x=target и c внутри A_ij."""
    old_mid = (a_lo_row + a_hi_row) / 2.0
    eps = 1e-9 * np.maximum(1.0, a_hi_row - a_lo_row)
    lower = a_lo_row + eps
    upper = a_hi_row - eps
    x1, x2 = float(x[0]), float(x[1])

    if abs(x1) < 1e-12 and abs(x2) < 1e-12:
        raise ValueError("Нельзя выбрать центры матрицы для нулевого вектора x.")

    if abs(x1) < 1e-12:
        c2 = target / x2
        if not (lower[1] <= c2 <= upper[1]):
            raise ValueError("mid(b_i) недостижима при заданном x внутри исходных A_ij.")
        return np.array([old_mid[0], c2], dtype=float)

    if abs(x2) < 1e-12:
        c1 = target / x1
        if not (lower[0] <= c1 <= upper[0]):
            raise ValueError("mid(b_i) недостижима при заданном x внутри исходных A_ij.")
        return np.array([c1, old_mid[1]], dtype=float)

    # Из c1*x1+c2*x2=target выражаем c2 через c1 и учитываем границы.
    c1_from_c2_lower = (target - x2 * lower[1]) / x1
    c1_from_c2_upper = (target - x2 * upper[1]) / x1
    feasible_c1_low = max(lower[0], min(c1_from_c2_lower, c1_from_c2_upper))
    feasible_c1_high = min(upper[0], max(c1_from_c2_lower, c1_from_c2_upper))

    if feasible_c1_low > feasible_c1_high + 1e-10:
        raise ValueError(
            "mid(b_i) не представима как c_i*x при c_i внутри исходных интервалов A."
        )

    # Минимизируем ||c-mid(A_i)||_2 при линейном ограничении c*x=target.
    slope = x1 / x2
    intercept = target / x2
    unconstrained_c1 = (
        old_mid[0] + slope * (intercept - old_mid[1])
    ) / (1.0 + slope * slope)
    c1 = float(np.clip(unconstrained_c1, feasible_c1_low, feasible_c1_high))
    c2 = (target - x1 * c1) / x2
    return np.array([c1, c2], dtype=float)


def shifted_subinterval_a_correction(
    system: IntervalSystem,
    x_reference: np.ndarray,
    target_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Обобщённая A-коррекция с выбором суженных подынтервалов внутри A.

    Для каждой строки выбирается центр c_i, удовлетворяющий c_i*x=mid(b_i).
    Затем радиусы новых интервалов ограничиваются условием
        sum_j rad(A'_ij)*|x_j| <= rad(b_i)-target_margin.
    Поэтому Tol(x,A',b)>=target_margin.
    """
    mid_b, rad_b = interval_mid_rad(system.b_lo, system.b_hi)
    centers = np.vstack([
        closest_row_center_inside_original(
            system.a_lo[i], system.a_hi[i], float(mid_b[i]), x_reference
        )
        for i in range(system.a_lo.shape[0])
    ])

    # Максимальные симметричные радиусы, при которых A'_ij остаётся
    # подынтервалом исходного A_ij.
    available_radii = np.minimum(centers - system.a_lo, system.a_hi - centers)
    available_radii = np.maximum(available_radii, 0.0)
    new_radii = np.zeros_like(available_radii)

    for i in range(system.a_lo.shape[0]):
        budget = float(rad_b[i] - target_margin)
        if budget < -1e-12:
            raise ValueError(
                "Запас коррекции больше радиуса b_i; уменьшите CORRECTION_MARGIN."
            )
        budget = max(0.0, budget)
        current_uncertainty = float(np.sum(available_radii[i] * np.abs(x_reference)))

        if current_uncertainty <= budget or current_uncertainty <= 1e-15:
            new_radii[i] = available_radii[i]
        else:
            new_radii[i] = available_radii[i] * (budget / current_uncertainty)

    new_a_lo = centers - new_radii
    new_a_hi = centers + new_radii

    # Контроль: полученные интервалы должны быть подмножествами исходных.
    if np.any(new_a_lo < system.a_lo - 1e-8) or np.any(new_a_hi > system.a_hi + 1e-8):
        raise RuntimeError("Полученные интервалы A' вышли за границы исходных A.")

    return new_a_lo, new_a_hi, centers


def correct_a(system: IntervalSystem, original_argmax: np.ndarray) -> tuple[IntervalSystem, str, dict[str, float]]:
    corrected = system.copy(name=f"{system.name}_A-correction")
    standard = find_symmetric_a_correction(system, CORRECTION_MARGIN)

    if standard is not None:
        corrected.a_lo, corrected.a_hi, alpha, achieved = standard
        mode = "симметричное сужение с сохранением mid(A)"
        details = {
            "alpha": float(alpha),
            "achieved": float(achieved),
            "max_center_shift": 0.0,
        }
    else:
        # В таком случае точечная матрица mid(A) уже несовместна.
        # Сохранение её середин не позволит добиться Tol>=0 одним сужением.
        corrected.a_lo, corrected.a_hi, centers = shifted_subinterval_a_correction(
            system, original_argmax, CORRECTION_MARGIN
        )
        achieved = maximize_tol(corrected).value
        old_centers, _ = interval_mid_rad(system.a_lo, system.a_hi)
        max_shift = float(np.max(np.abs(centers - old_centers)))
        mode = "сдвиг середин внутри исходных A_ij + сужение подынтервалов"
        details = {
            "alpha": float("nan"),
            "achieved": float(achieved),
            "max_center_shift": max_shift,
        }

    if details["achieved"] < -1e-7:
        raise RuntimeError(f"A-коррекция для {system.name} не обеспечила Tol>=0.")

    return corrected, mode, details


# Ab-коррекция: одновременно уширяем b и сужаем A.
# Используется одинаковый принцип симметричного сужения A, но уже для
# расширенной правой части. Это соответствует комбинированной стратегии §1.6.4.


def correct_ab(system: IntervalSystem, original_max: float) -> tuple[IntervalSystem, float, float]:
    if original_max >= 0.0:
        return system.copy(name=f"{system.name}_Ab-correction"), 0.0, 1.0

    required_increase = -original_max
    k_b = AB_B_SHARE * required_increase

    corrected = correct_b_uniform(
        system,
        k=k_b,
        name=f"{system.name}_Ab-correction",
    )

    symmetric = find_symmetric_a_correction(corrected, CORRECTION_MARGIN)
    if symmetric is None:
        raise RuntimeError(
            f"Для Ab-коррекции {system.name} не удалось подобрать симметричное сужение A."
        )

    corrected.a_lo, corrected.a_hi, alpha, _ = symmetric
    final_max = maximize_tol(corrected).value
    if final_max < -1e-7:
        raise RuntimeError(f"Ab-коррекция для {system.name} не обеспечила Tol>=0.")

    return corrected, k_b, alpha


# -----------------------------------------------------------------------------
# Сравнение искажений исходных интервальных данных.
# Используется относительная евклидова норма изменений середин и радиусов.
# -----------------------------------------------------------------------------


def relative_distortion(
    original: IntervalSystem,
    corrected: IntervalSystem,
) -> tuple[float, float, float]:
    mid_a, rad_a = interval_mid_rad(original.a_lo, original.a_hi)
    mid_b, rad_b = interval_mid_rad(original.b_lo, original.b_hi)
    new_mid_a, new_rad_a = interval_mid_rad(corrected.a_lo, corrected.a_hi)
    new_mid_b, new_rad_b = interval_mid_rad(corrected.b_lo, corrected.b_hi)

    delta_a = np.sqrt(
        np.sum((new_mid_a - mid_a) ** 2) + np.sum((new_rad_a - rad_a) ** 2)
    )
    norm_a = np.sqrt(np.sum(mid_a ** 2) + np.sum(rad_a ** 2))

    delta_b = np.sqrt(
        np.sum((new_mid_b - mid_b) ** 2) + np.sum((new_rad_b - rad_b) ** 2)
    )
    norm_b = np.sqrt(np.sum(mid_b ** 2) + np.sum(rad_b ** 2))

    rel_a = float(delta_a / norm_a) if norm_a > 0 else 0.0
    rel_b = float(delta_b / norm_b) if norm_b > 0 else 0.0
    rel_total = float(
        np.sqrt(delta_a ** 2 + delta_b ** 2) /
        np.sqrt(norm_a ** 2 + norm_b ** 2)
    )
    return rel_a, rel_b, rel_total


# -----------------------------------------------------------------------------
# Графики: 3D-поверхность Tol и плоскость допускового множества Tol>=0.
# -----------------------------------------------------------------------------


def tol_grid(
    system: IntervalSystem,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
) -> np.ndarray:
    mid_b, rad_b = interval_mid_rad(system.b_lo, system.b_hi)
    result = np.full_like(x1_grid, np.inf, dtype=float)

    for i in range(system.a_lo.shape[0]):
        lower_ax = np.zeros_like(x1_grid, dtype=float)
        upper_ax = np.zeros_like(x1_grid, dtype=float)

        for j, x_grid in enumerate((x1_grid, x2_grid)):
            p1 = system.a_lo[i, j] * x_grid
            p2 = system.a_hi[i, j] * x_grid
            lower_ax += np.minimum(p1, p2)
            upper_ax += np.maximum(p1, p2)

        row_value = rad_b[i] - np.maximum(
            np.abs(mid_b[i] - lower_ax),
            np.abs(mid_b[i] - upper_ax),
        )
        result = np.minimum(result, row_value)

    return result


def plot_system(
    system: IntervalSystem,
    maximum: MaximumResult,
    path: Path,
) -> None:
    x_star = maximum.x
    span_x1 = max(2.0, 0.75 * abs(float(x_star[0])) + 1.0)
    span_x2 = max(2.0, 0.55 * abs(float(x_star[1])) + 1.0)

    # Для 3D-графика используем широкий диапазон, чтобы показать общую форму Tol.
    x1 = np.linspace(x_star[0] - span_x1, x_star[0] + span_x1, GRID_SIZE)
    x2 = np.linspace(x_star[1] - span_x2, x_star[1] + span_x2, GRID_SIZE)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    z = tol_grid(system, x1_grid, x2_grid)

    # Допусковое множество после коррекции может быть очень маленьким.
    # Поэтому контурный график строится в увеличенном масштабе вокруг argmax.
    set_span_x1 = max(0.10, 0.05 * span_x1)
    set_span_x2 = max(0.10, 0.05 * span_x2)
    x1_set = np.linspace(
        x_star[0] - set_span_x1, x_star[0] + set_span_x1, FEASIBLE_GRID_SIZE
    )
    x2_set = np.linspace(
        x_star[1] - set_span_x2, x_star[1] + set_span_x2, FEASIBLE_GRID_SIZE
    )
    x1_set_grid, x2_set_grid = np.meshgrid(x1_set, x2_set)
    z_set = tol_grid(system, x1_set_grid, x2_set_grid)

    fig = plt.figure(figsize=(13, 5.5))
    ax_surface = fig.add_subplot(1, 2, 1, projection="3d")
    surface = ax_surface.plot_surface(
        x1_grid,
        x2_grid,
        z,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    ax_surface.scatter(
        [x_star[0]], [x_star[1]], [maximum.value],
        color="red", s=55, marker="o", label="argmax Tol",
    )
    ax_surface.set_title("Распознающий функционал Tol(x)")
    ax_surface.set_xlabel("x1")
    ax_surface.set_ylabel("x2")
    ax_surface.set_zlabel("Tol")
    ax_surface.legend(loc="upper left")
    fig.colorbar(surface, ax=ax_surface, shrink=0.62, pad=0.08)

    ax_set = fig.add_subplot(1, 2, 2)
    contour = ax_set.contourf(
        x1_set_grid, x2_set_grid, z_set, levels=35, cmap="viridis"
    )
    if np.nanmax(z_set) > 0.0:
        # Отдельно закрашиваем именно допусковое множество, чтобы небольшая
        # область Tol>=0 не терялась среди уровней всего отрицательного фона.
        ax_set.contourf(
            x1_set_grid, x2_set_grid, z_set,
            levels=[0.0, float(np.nanmax(z_set)) + 1e-12],
            colors=["#ff5252"], alpha=0.85,
        )
        ax_set.contour(
            x1_set_grid, x2_set_grid, z_set,
            levels=[0.0], colors="black", linewidths=1.5,
        )
    ax_set.scatter(
        [x_star[0]], [x_star[1]],
        color="red", marker="x", s=75, linewidths=2,
        label=f"argmax Tol = {maximum.value:.4f}",
    )
    ax_set.set_title("Допусковое множество: Tol(x) ≥ 0")
    ax_set.set_xlabel("x1")
    ax_set.set_ylabel("x2")
    ax_set.legend(loc="best")
    ax_set.grid(alpha=0.2)
    fig.colorbar(contour, ax=ax_set, label="Tol")

    fig.suptitle(
        f"{system.name}: max Tol = {maximum.value:.6f}; "
        f"argmax = ({x_star[0]:.4f}, {x_star[1]:.4f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    if SAVE_PLOTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"График сохранён: {path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# Главный расчёт.
# -----------------------------------------------------------------------------


def main() -> None:
    print("Лабораторная работа №3 — интервальный анализ, вариант 5")
    print("Максимизация Tol выполняется линейным программированием по квадрантам.")
    print(f"Положительный запас после коррекции: {CORRECTION_MARGIN:g}\n")

    summary: list[tuple[str, float, float, float, float, str]] = []

    for original in SYSTEMS:
        original_max = print_analysis(original)

        if original_max.value >= 0.0:
            print(f"{original.name}: исходная система уже разрешима; коррекции не требуются.")
            continue

        # 1. b-коррекция: K = -T + margin.
        k_b = -original_max.value + CORRECTION_MARGIN
        b_corrected = correct_b_uniform(
            original,
            k=k_b,
            name=f"{original.name}_b-correction",
        )
        b_max = print_analysis(b_corrected)
        print(f"b-коррекция: K = {k_b:.9f}; теоретически max Tol увеличивается на K.")

        # 2. A-коррекция: сначала метод с неизменной серединой.
        mid_a, _ = interval_mid_rad(original.a_lo, original.a_hi)
        point_matrix = original.copy(name=f"{original.name}_point_matrix")
        point_matrix.a_lo = mid_a.copy()
        point_matrix.a_hi = mid_a.copy()
        point_matrix_max = maximize_tol(point_matrix).value

        a_corrected, a_mode, a_details = correct_a(original, original_max.x)
        a_max = print_analysis(a_corrected)
        print(f"A-коррекция: {a_mode}.")
        print(f"max Tol для точечной mid(A): {point_matrix_max:.9f}.")
        if np.isfinite(a_details["alpha"]):
            print(
                "Остаточная доля радиусов A: "
                f"alpha={a_details['alpha']:.9f}; "
                f"сужение={(1.0 - a_details['alpha']) * 100:.3f}% ."
            )
        else:
            print(
                "Максимальный сдвиг середин A относительно исходных: "
                f"{a_details['max_center_shift']:.9f}."
            )

        # 3. Ab-коррекция: b расширяется, а радиусы A одновременно уменьшаются.
        ab_corrected, k_ab, alpha_ab = correct_ab(original, original_max.value)
        ab_max = print_analysis(ab_corrected)
        print(
            "Ab-коррекция: "
            f"K_b={k_ab:.9f}; остаточная доля радиусов A alpha={alpha_ab:.9f}."
        )

        # Сравнение относительного изменения интервалов.
        d_a = relative_distortion(original, a_corrected)
        d_b = relative_distortion(original, b_corrected)
        d_ab = relative_distortion(original, ab_corrected)

        print("\nОтносительное искажение исходных данных:")
        print("  Коррекция       dA         db         d_total")
        print(f"  A           {d_a[0]:.6f}   {d_a[1]:.6f}   {d_a[2]:.6f}")
        print(f"  b           {d_b[0]:.6f}   {d_b[1]:.6f}   {d_b[2]:.6f}")
        print(f"  Ab          {d_ab[0]:.6f}   {d_ab[1]:.6f}   {d_ab[2]:.6f}")

        distortions = {"A": d_a[2], "b": d_b[2], "Ab": d_ab[2]}
        least_distorting = min(distortions, key=distortions.get)
        print(
            "Наименьшее относительное искажение по выбранной норме: "
            f"{least_distorting} ({distortions[least_distorting]:.6f})."
        )

        summary.append((
            original.name,
            original_max.value,
            a_max.value,
            b_max.value,
            ab_max.value,
            least_distorting,
        ))

    print("\n" + "=" * 76)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 76)
    print("Система      T исходн.       T после A      T после b      T после Ab     минимум искажения")
    for name, t0, ta, tb, tab, least in summary:
        print(
            f"{name:<8} {t0:>12.6f} {ta:>14.6f} {tb:>14.6f} "
            f"{tab:>14.6f} {least:>16}"
        )

    if SAVE_PLOTS:
        print(f"\nВсе графики находятся в папке: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
