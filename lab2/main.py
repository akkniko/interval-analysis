import math
import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def down(value):
    return math.nextafter(float(value), -math.inf)


def up(value):
    return math.nextafter(float(value), math.inf)


def as_interval(value):
    if isinstance(value, Interval):
        return value
    return Interval(float(value), float(value))


@dataclass
class Interval:
    low: float
    high: float

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError("The lower bound cannot be greater than the upper bound")

    def __add__(self, other):
        other = as_interval(other)
        return Interval(down(self.low + other.low), up(self.high + other.high))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = as_interval(other)
        return Interval(down(self.low - other.high), up(self.high - other.low))

    def __rsub__(self, other):
        return as_interval(other) - self

    def __mul__(self, other):
        other = as_interval(other)
        products = [
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high,
        ]
        return Interval(down(min(products)), up(max(products)))

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Only non-negative integer powers are supported")

        if power == 0:
            return Interval(1.0, 1.0)

        if power % 2 == 0:
            values = [self.low ** power, self.high ** power]
            lower = 0.0 if self.low <= 0 <= self.high else min(values)
            return Interval(down(lower), up(max(values)))

        return Interval(down(self.low ** power), up(self.high ** power))

    def midpoint(self):
        return (self.low + self.high) / 2.0

    def radius(self):
        return (self.high - self.low) / 2.0

    def __str__(self):
        return f"[{self.low:.6f}, {self.high:.6f}]"


def interval_sin(value):
    if value.high - value.low >= 2 * math.pi:
        return Interval(-1.0, 1.0)

    values = [math.sin(value.low), math.sin(value.high)]

    first_max = math.ceil((value.low - math.pi / 2) / (2 * math.pi))
    last_max = math.floor((value.high - math.pi / 2) / (2 * math.pi))
    if first_max <= last_max:
        values.append(1.0)

    first_min = math.ceil((value.low + math.pi / 2) / (2 * math.pi))
    last_min = math.floor((value.high + math.pi / 2) / (2 * math.pi))
    if first_min <= last_min:
        values.append(-1.0)

    return Interval(down(min(values)), up(max(values)))


def interval_cos(value):
    if value.high - value.low >= 2 * math.pi:
        return Interval(-1.0, 1.0)

    values = [math.cos(value.low), math.cos(value.high)]

    first_max = math.ceil(value.low / (2 * math.pi))
    last_max = math.floor(value.high / (2 * math.pi))
    if first_max <= last_max:
        values.append(1.0)

    first_min = math.ceil((value.low - math.pi) / (2 * math.pi))
    last_min = math.floor((value.high - math.pi) / (2 * math.pi))
    if first_min <= last_min:
        values.append(-1.0)

    return Interval(down(min(values)), up(max(values)))


def sinc_value(value):
    if abs(value) < 1e-12:
        return 1.0
    return math.sin(value) / value


def interval_sinc(value):

    candidates = [sinc_value(value.low), sinc_value(value.high)]
    stationary_points = [-4.493409457909064, 0.0, 4.493409457909064]

    for point in stationary_points:
        if value.low <= point <= value.high:
            candidates.append(sinc_value(point))

    return Interval(down(min(candidates)), up(max(candidates)))


def interval_hull(intervals):
    return Interval(
        min(item.low for item in intervals),
        max(item.high for item in intervals),
    )


def interval_intersection(first, second):
    low = max(first.low, second.low)
    high = min(first.high, second.high)
    if low > high:
        raise ValueError("The two interval estimates do not intersect")
    return Interval(low, high)


def hausdorff_distance(first, second):
    return max(abs(first.low - second.low), abs(first.high - second.high))


def clipped(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def f1(x):
    return x ** 3 - 3 * x ** 2 + 2


def f1_derivative(x):
    return 3 * x ** 2 - 6 * x


def f1_interval_original(x):
    return x ** 3 - 3 * (x ** 2) + 2


def f1_interval_horner(x):
    return ((x - 3) * x) * x + 2


def f1_derivative_interval(x):
    points = [x.low, x.high]
    if x.low <= 1.0 <= x.high:
        points.append(1.0)
    values = [f1_derivative(point) for point in points]
    return Interval(down(min(values)), up(max(values)))


def f1_slope_interval(x, center):
    return (
        x ** 2
        + (center - 3) * x
        + center * (center - 3)
    )


def f2(x):
    return x ** 2 + np.cos(3 * x)


def f2_derivative(x):
    return 2 * x - 3 * np.sin(3 * x)


def f2_interval_original(x):
    return x ** 2 + interval_cos(3 * x)


def f2_interval_equivalent(x):
    sine = interval_sin(1.5 * x)
    return x ** 2 + 1 - 2 * (sine ** 2)


def f2_derivative_interval(x):
    critical_points = find_roots(
        lambda point: 2 - 9 * math.cos(3 * point), x.low, x.high
    )
    points = [x.low, x.high] + critical_points
    values = [f2_derivative(point) for point in points]
    return Interval(down(min(values)), up(max(values)))


def f2_slope_interval(x, center):
    first_part = x + center
    sine_part = interval_sin(1.5 * (x + center))
    sinc_part = interval_sinc(1.5 * (x - center))
    return first_part - 3 * sine_part * sinc_part


def f3(x, y):
    return 2 * x ** 2 + 3 * y ** 2 - 2 * x * y + x - 2 * y


def f3_interval_original(x, y):
    return 2 * (x ** 2) + 3 * (y ** 2) - 2 * x * y + x - 2 * y


def f3_interval_equivalent(x, y):
    return (
        -0.35
        + 2 * (x - 0.5 * y + 0.25) ** 2
        + 2.5 * (y - 0.3) ** 2
    )


def f3_dx_interval(x, y):
    return 4 * x - 2 * y + 1


def f3_dy_interval(x, y):
    return 6 * y - 2 * x - 2

def mean_value_1d(function, derivative_interval, x_interval, center):
    return as_interval(function(center)) + derivative_interval(x_interval) * (x_interval - center)


def slope_form_1d(function, slope_interval, x_interval, center):
    return as_interval(function(center)) + slope_interval(x_interval, center) * (x_interval - center)


def baumann_centers(x_interval, derivative_interval):
    derivative = derivative_interval(x_interval)
    if derivative.radius() == 0:
        p = 0.0 if derivative.midpoint() == 0 else math.copysign(1.0, derivative.midpoint())
    else:
        p = clipped(derivative.midpoint() / derivative.radius())

    center = x_interval.midpoint()
    radius = x_interval.radius()
    return center - p * radius, center + p * radius


def bicentered_form_1d(function, derivative_interval, x_interval):
    first_center, second_center = baumann_centers(x_interval, derivative_interval)
    first = mean_value_1d(function, derivative_interval, x_interval, first_center)
    second = mean_value_1d(function, derivative_interval, x_interval, second_center)
    return interval_intersection(first, second), first_center, second_center, first, second


def mean_value_2d(x_interval, y_interval, center):
    cx, cy = center
    value_at_center = f3(cx, cy)
    dx = f3_dx_interval(x_interval, y_interval)
    dy = f3_dy_interval(x_interval, y_interval)
    return (
        as_interval(value_at_center)
        + dx * (x_interval - cx)
        + dy * (y_interval - cy)
    )


def bicentered_form_2d(x_interval, y_interval):
    dx = f3_dx_interval(x_interval, y_interval)
    dy = f3_dy_interval(x_interval, y_interval)

    px = clipped(dx.midpoint() / dx.radius()) if dx.radius() else 0.0
    py = clipped(dy.midpoint() / dy.radius()) if dy.radius() else 0.0

    mx, my = x_interval.midpoint(), y_interval.midpoint()
    rx, ry = x_interval.radius(), y_interval.radius()

    first_center = (mx - px * rx, my - py * ry)
    second_center = (mx + px * rx, my + py * ry)

    first = mean_value_2d(x_interval, y_interval, first_center)
    second = mean_value_2d(x_interval, y_interval, second_center)
    result = interval_intersection(first, second)

    return result, first_center, second_center, first, second

def bisect_root(function, left, right):
    f_left = function(left)
    f_right = function(right)

    if f_left == 0:
        return left
    if f_right == 0:
        return right
    if f_left * f_right > 0:
        return None

    for _ in range(80):
        middle = (left + right) / 2
        f_middle = function(middle)

        if abs(f_middle) < 1e-13:
            return middle
        if f_left * f_middle <= 0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle

    return (left + right) / 2


def find_roots(function, left, right, steps=10000):
    grid = np.linspace(left, right, steps + 1)
    values = [function(point) for point in grid]
    roots = []

    for i in range(steps):
        if values[i] == 0:
            roots.append(float(grid[i]))
        elif values[i] * values[i + 1] < 0:
            root = bisect_root(function, float(grid[i]), float(grid[i + 1]))
            if root is not None:
                roots.append(root)

    if values[-1] == 0:
        roots.append(float(grid[-1]))

    unique_roots = []
    for root in roots:
        if not unique_roots or abs(root - unique_roots[-1]) > 1e-7:
            unique_roots.append(root)
    return unique_roots


def numerical_range_1d(function, derivative, left, right):
    critical_points = find_roots(derivative, left, right)
    points = [left, right] + critical_points
    values = [function(point) for point in points]
    return Interval(min(values), max(values)), critical_points


def exact_range_f1():
    points = [0.0, 2.0, 3.0]
    values = [f1(point) for point in points]
    return Interval(min(values), max(values))


def exact_range_f3():
    minimum = f3(-0.1, 0.3)

    corners = [
        (-2.0, -1.0), (-2.0, 2.0),
        (1.0, -1.0), (1.0, 2.0),
    ]
    maximum = max(f3(x, y) for x, y in corners)
    return Interval(minimum, maximum)

def print_estimate(name, estimate, exact_range):
    print(f"{name:<39} {str(estimate):<25} Hausdorff = {hausdorff_distance(estimate, exact_range):.6f}")


def analyze_one_variable(name, function, derivative_interval, natural_function,
                         equivalent_function, slope_function, x_interval,
                         exact_range, lipschitz_constant):
    print("\n" + "=" * 78)
    print(name)
    print("Interval:", x_interval)
    print("Exact/numerical range:", exact_range)

    natural = natural_function(x_interval)
    equivalent = equivalent_function(x_interval)
    print_estimate("B.1 Natural expression", natural, exact_range)
    print_estimate("B.2 Equivalent expression", equivalent, exact_range)

    centers = [x_interval.low, x_interval.midpoint(), x_interval.high]
    for center in centers:
        estimate = mean_value_1d(function, derivative_interval, x_interval, center)
        print_estimate(f"B.3 Differential form, c={center:.3f}", estimate, exact_range)

    for center in centers:
        estimate = slope_form_1d(function, slope_function, x_interval, center)
        print_estimate(f"B.4 Slope form, c={center:.3f}", estimate, exact_range)

    bicentered, c1, c2, first, second = bicentered_form_1d(
        function, derivative_interval, x_interval
    )
    print_estimate("B.5 Baumann bicentered form", bicentered, exact_range)
    print(f"    Baumann centers: c1={c1:.6f}, c2={c2:.6f}")
    print(f"    First centered interval:  {first}")
    print(f"    Second centered interval: {second}")

    theoretical_bound = lipschitz_constant * x_interval.radius()
    print("\nLipschitz analysis of the natural interval extension:")
    print(f"L = {lipschitz_constant:.6f}")
    print(f"rad(F(X)) = {natural.radius():.6f}")
    print(f"L * rad(X) = {theoretical_bound:.6f}")
    print("Inequality holds:", natural.radius() <= theoretical_bound + 1e-10)


def plot_one_variable(function, interval, title, filename):
    x = np.linspace(interval.low, interval.high, 600)
    y = function(x)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, label=title)
    plt.axhline(0, linewidth=0.8)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=160)
    plt.close()


def plot_f3(x_interval, y_interval):
    x_values = np.linspace(x_interval.low, x_interval.high, 180)
    y_values = np.linspace(y_interval.low, y_interval.high, 180)
    xx, yy = np.meshgrid(x_values, y_values)
    zz = f3(xx, yy)

    figure = plt.figure(figsize=(9, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none", alpha=0.9)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("f3(x, y)")
    axis.set_title("Surface of f3(x, y)")
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT_DIR, "f3_surface.png"), dpi=160)
    plt.close(figure)

    plt.figure(figsize=(7, 6))
    filled = plt.contourf(xx, yy, zz, levels=25, cmap="viridis")
    lines = plt.contour(xx, yy, zz, levels=15, colors="black", linewidths=0.6)
    plt.clabel(lines, inline=True, fontsize=8)
    plt.colorbar(filled, label="f3(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Contour lines of f3(x, y)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "f3_contours.png"), dpi=160)
    plt.close()

def main():
    x1 = Interval(0.0, 3.0)
    x2 = Interval(-2.0, 1.0)
    x3 = Interval(-2.0, 1.0)
    y3 = Interval(-1.0, 2.0)

    range1 = exact_range_f1()
    range2, critical2 = numerical_range_1d(f2, f2_derivative, x2.low, x2.high)

    analyze_one_variable(
        "f1(x) = x^3 - 3x^2 + 2",
        f1,
        f1_derivative_interval,
        f1_interval_original,
        f1_interval_horner,
        f1_slope_interval,
        x1,
        range1,
        lipschitz_constant=45.0,
    )

    analyze_one_variable(
        "f2(x) = x^2 + cos(3x)",
        f2,
        f2_derivative_interval,
        f2_interval_original,
        f2_interval_equivalent,
        f2_slope_interval,
        x2,
        range2,
        lipschitz_constant=7.0,
    )

    print("\nCritical points of f2 on [-2, 1]:")
    for point in critical2:
        print(f"x = {point:.8f}, f2(x) = {f2(point):.8f}")

    print("\n" + "=" * 78)
    print("f3(x, y) = 2x^2 + 3y^2 - 2xy + x - 2y")
    print("X = [-2, 1] x [-1, 2]")

    exact3 = exact_range_f3()
    natural3 = f3_interval_original(x3, y3)
    equivalent3 = f3_interval_equivalent(x3, y3)

    print("Exact range:", exact3)
    print_estimate("E.3 Natural expression", natural3, exact3)
    print_estimate("E.4 Equivalent expression", equivalent3, exact3)

    midpoint = (x3.midpoint(), y3.midpoint())
    other_center = (-1.0, 1.0)
    centered_mid = mean_value_2d(x3, y3, midpoint)
    centered_other = mean_value_2d(x3, y3, other_center)

    print_estimate(f"E.5 Differential form, c={midpoint}", centered_mid, exact3)
    print_estimate(f"E.5 Differential form, c={other_center}", centered_other, exact3)

    bicentered, c1, c2, first, second = bicentered_form_2d(x3, y3)
    print_estimate("E.6 Baumann bicentered form", bicentered, exact3)
    print("    Baumann center 1:", c1)
    print("    Baumann center 2:", c2)
    print("    First centered interval:", first)
    print("    Second centered interval:", second)

    x_mid = x3.midpoint()
    y_mid = y3.midpoint()

    split_x = [
        f3_interval_original(Interval(x3.low, x_mid), y3),
        f3_interval_original(Interval(x_mid, x3.high), y3),
    ]
    hull_x = interval_hull(split_x)

    split_y = [
        f3_interval_original(x3, Interval(y3.low, y_mid)),
        f3_interval_original(x3, Interval(y_mid, y3.high)),
    ]
    hull_y = interval_hull(split_y)

    print_estimate("E.7 Hull after split by x", hull_x, exact3)
    print_estimate("E.7 Hull after split by y", hull_y, exact3)
    print("    Subintervals after split by x:", split_x)
    print("    Subintervals after split by y:", split_y)

    dx_interval = f3_dx_interval(x3, y3)
    dy_interval = f3_dy_interval(x3, y3)
    lipschitz_x = max(abs(dx_interval.low), abs(dx_interval.high))
    lipschitz_y = max(abs(dy_interval.low), abs(dy_interval.high))
    theoretical_3 = lipschitz_x * x3.radius() + lipschitz_y * y3.radius()

    print("\nE.8 Coordinate Lipschitz estimates:")
    print("df3/dx interval:", dx_interval, "=> L1 =", lipschitz_x)
    print("df3/dy interval:", dy_interval, "=> L2 =", lipschitz_y)
    print(f"rad(F3(X)) = {natural3.radius():.6f}")
    print(f"L1*rad(X1) + L2*rad(X2) = {theoretical_3:.6f}")
    print("Inequality holds:", natural3.radius() <= theoretical_3 + 1e-10)

    print("\nE.9 Comparison of all estimates for f3:")
    estimates = [
        ("Natural expression", natural3),
        ("Equivalent expression", equivalent3),
        ("Differential form at midpoint", centered_mid),
        ("Differential form at another center", centered_other),
        ("Baumann bicentered form", bicentered),
        ("Split by x", hull_x),
        ("Split by y", hull_y),
    ]
    for name, estimate in estimates:
        print_estimate(name, estimate, exact3)

    plot_one_variable(f1, x1, "f1(x) = x^3 - 3x^2 + 2", "f1_plot.png")
    plot_one_variable(f2, x2, "f2(x) = x^2 + cos(3x)", "f2_plot.png")
    plot_f3(x3, y3)

    print("\nPlots were saved:")
    print("  f1_plot.png")
    print("  f2_plot.png")
    print("  f3_surface.png")
    print("  f3_contours.png")


if __name__ == "__main__":
    main()