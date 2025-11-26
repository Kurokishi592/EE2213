"""Graphical method LP solver for TWO decision variables.

Features:
1. Accept linear objective and linear constraints with two variables (x, y).
2. Plot constraint lines; shade each half-space; highlight feasible region.
3. Compute all feasible vertices (pairwise intersections + axes if non-negativity).
4. Draw objective level (iso-profit) lines and identify optimal solution by sliding
   in direction of optimization (max or min).
5. Classify outcome:
      - unique optimal solution
      - infinite optimal solutions (objective constant along an edge)
      - unbounded (objective can increase/decrease without limit within feasible region)
      - infeasible (no feasible region)

Input format (example):
    objective = (c1, c2)  # coefficients for z = c1*x + c2*y
    sense = 'max' or 'min'
    constraints = [
        (a1, b1, '<=', rhs1),
        (a2, b2, '>=', rhs2),
        (a3, b3, '=', rhs3),  # optional equality
        ...
    ]
    nonneg = True  # add x >= 0, y >= 0 automatically

Return structure (dict):
    {
        'status': 'optimal' | 'infinite' | 'unbounded' | 'infeasible',
        'optimal_value': float or None,
        'optimal_points': [(x, y), ...] or None,
        'vertices': [(x, y), ...],
        'objective': (c1, c2),
        'sense': sense
    }

NOTE: For classification we rely purely on geometry, not external LP solvers.
Unbounded detection is done heuristically; for two variables typical classroom
examples will work correctly.
"""
from __future__ import annotations
import numpy as np
import itertools
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Sequence

Constraint = Tuple[float, float, str, float]  # (a, b, op, rhs)

# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def _normalize_constraint(a: float, b: float, op: str, rhs: float) -> Tuple[float, float, float]:
    """Convert constraint to standard <= form: a*x + b*y <= c.
    For '>=', multiply both sides by -1. For '=' keep both forms (handled separately)."""
    if op == '<=':
        return a, b, rhs
    elif op == '>=':
        return -a, -b, -rhs
    elif op == '=':
        # Equality will be represented as both <= and >=: handled by caller
        return a, b, rhs
    else:
        raise ValueError(f"Unknown operator: {op}")


def _generate_bounds(constraints: List[Constraint], nonneg: bool) -> Tuple[float, float, float, float]:
    """Heuristic axis bounds based on intercepts and provided constraints."""
    xs = []
    ys = []
    for a, b, op, c in constraints:
        if a != 0:
            xs.append(c / a)
        if b != 0:
            ys.append(c / b)
    if nonneg:
        xs.append(0); ys.append(0)
    # Remove infinities / invalid
    xs = [v for v in xs if np.isfinite(v)]
    ys = [v for v in ys if np.isfinite(v)]
    x_max = max(xs + [1]) if xs else 10
    y_max = max(ys + [1]) if ys else 10
    # Provide some margin
    return 0, max(1, x_max * 1.2), 0, max(1, y_max * 1.2)


def _intersection(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
    """Intersection of two lines given in normalized form (a,b,c) representing a*x + b*y = c."""
    a1, b1, c1v = c1
    a2, b2, c2v = c2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    x = (c1v * b2 - c2v * b1) / det
    y = (a1 * c2v - a2 * c1v) / det
    return x, y


def _is_feasible_point(x: float, y: float, constraints: List[Constraint], nonneg: bool) -> bool:
    eps = 1e-9
    if nonneg and (x < -eps or y < -eps):
        return False
    for a, b, op, c in constraints:
        val = a * x + b * y
        if op == '<=' and val - c > eps:
            return False
        if op == '>=' and c - val > eps:
            return False
        if op == '=' and abs(val - c) > eps:
            return False
    return True


def _objective_value(c: Tuple[float, float], x: float, y: float) -> float:
    return c[0] * x + c[1] * y


def _build_equality_pairs(constraints: List[Constraint]) -> List[Tuple[Tuple[float,float,float], Tuple[float,float,float]]]:
    pairs = []
    for a, b, op, c in constraints:
        if op == '=':
            # Represent equality as both <= and >= in normalized form
            pairs.append(((a, b, c), (-a, -b, -c)))
    return pairs


def _all_line_forms(constraints: List[Constraint], nonneg: bool) -> List[Tuple[float,float,float]]:
    """Return list of (a,b,c) representing lines a*x + b*y = c for each constraint (including axis if nonneg)."""
    forms = []
    for a, b, op, c in constraints:
        if op in ('<=', '>=', '='):
            forms.append((a, b, c))
    if nonneg:
        # x >= 0 -> line x = 0 -> 1*x + 0*y = 0
        forms.append((1, 0, 0))
        # y >= 0 -> line y = 0 -> 0*x + 1*y = 0
        forms.append((0, 1, 0))
    return forms


def _compute_vertices(constraints: List[Constraint], nonneg: bool) -> List[Tuple[float,float]]:
    line_forms = _all_line_forms(constraints, nonneg)
    vertices = []
    for (i, lf1), (j, lf2) in itertools.combinations(enumerate(line_forms), 2):
        pt = _intersection(lf1, lf2)
        if pt is None:
            continue
        x, y = pt
        if _is_feasible_point(x, y, constraints, nonneg):
            vertices.append((round(x, 10), round(y, 10)))
    # Remove duplicates
    unique = []
    for v in vertices:
        if v not in unique:
            unique.append(v)
    return unique


def _classify(objective: Tuple[float,float], sense: str, vertices: List[Tuple[float,float]], constraints: List[Constraint], nonneg: bool) -> Tuple[str, Optional[float], Optional[List[Tuple[float,float]]]]:
    if not vertices:
        # Check feasibility by sampling grid (if no vertices but maybe feasible unbounded region)
        if not _any_feasible_point(constraints, nonneg):
            return 'infeasible', None, None
        # Region could be a single line or empty; treat as infeasible or infinite if objective aligns
        return 'unbounded', None, None
    vals = [ _objective_value(objective, x, y) for x,y in vertices ]
    if sense == 'max':
        best = max(vals)
    else:
        best = min(vals)
    best_vertices = [v for v, val in zip(vertices, vals) if abs(val - best) < 1e-9]
    # Infinite optimal solutions if objective is constant along an edge segment of feasible region.
    infinite = _detect_infinite_optimal(best_vertices, vertices, constraints, objective)
    if infinite:
        return 'infinite', best, infinite
    # Unbounded detection heuristic: sample rays along directions improving objective and check constraints.
    if _is_unbounded(objective, sense, constraints, nonneg, vertices):
        return 'unbounded', None, None
    return 'optimal', best, best_vertices


def _any_feasible_point(constraints: List[Constraint], nonneg: bool) -> bool:
    # Simple grid sample
    xs = np.linspace(0, 10, 11)
    ys = np.linspace(0, 10, 11)
    for x in xs:
        for y in ys:
            if _is_feasible_point(x, y, constraints, nonneg):
                return True
    return False


def _detect_infinite_optimal(best_vertices: List[Tuple[float,float]], all_vertices: List[Tuple[float,float]], constraints: List[Constraint], objective: Tuple[float,float]) -> Optional[List[Tuple[float,float]]]:
    if len(best_vertices) < 2:
        return None
    # Build edges along constraint lines: For each constraint line collect vertices lying on it.
    edges = []
    line_forms = _all_line_forms(constraints, True)  # include axes for potential edges
    for a,b,c in line_forms:
        on_line = [v for v in best_vertices if abs(a*v[0] + b*v[1] - c) < 1e-9]
        if len(on_line) >= 2:
            # Sort by projection for consistency
            d = np.array(on_line)
            # Use first non-zero direction
            if abs(a) > 1e-9:
                order = np.argsort(d[:,0])
            else:
                order = np.argsort(d[:,1])
            seq = [tuple(d[i]) for i in order]
            # Objective gradient
            grad = np.array(objective)
            # Check if gradient dot edge direction == 0 for any adjacent pair
            for p,q in zip(seq[:-1], seq[1:]):
                direction = np.array(q) - np.array(p)
                if abs(np.dot(grad, direction)) < 1e-9:
                    # Entire segment optimal (assume convexity)
                    return [p, q]
    return None


def _is_unbounded(objective: Tuple[float,float], sense: str, constraints: List[Constraint], nonneg: bool, vertices: List[Tuple[float,float]]) -> bool:
    # Heuristic: If no constraint limits growth in direction of objective gradient (for max) or negative gradient (for min).
    grad = np.array(objective, dtype=float)
    if sense == 'min':
        grad = -grad
    # For each constraint convert to normalized <= form and see if it provides an upper bound.
    provides_bound = False
    for a,b,op,c in constraints:
        # Norm direction of constraint outward normal for <= is (a,b).
        if op == '>=':
            # Flip sign so outward normal of feasible half-space points opposite
            a_n,b_n,c_n = -a,-b,-c
        else:
            a_n,b_n,c_n = a,b,c
        # If dot(grad, normal) > 0 then constraint faces the improving direction; it can bound the region.
        if a_n*grad[0] + b_n*grad[1] > 1e-9:
            provides_bound = True
            break
    if not provides_bound:
        return True
    return False

# ---------------------------------------------------------------
# Main solve and plot
# ---------------------------------------------------------------

def solve_lp_2var(objective: Tuple[float,float], sense: str, constraints: List[Constraint], nonneg: bool=True, plot: bool=True, show_objective_lines: bool=True):
    """Solve and plot 2-variable LP via graphical method."""
    if sense not in ('max','min'):
        raise ValueError("sense must be 'max' or 'min'")

    # Normalize operators in constraints (allow '=>' and '=<')
    normalized_constraints: List[Constraint] = []
    for a,b,op,c in constraints:
        if op == '=>':
            op = '>='
        elif op == '=<':
            op = '<='
        if op not in ('<=','>=','='):
            print(f"Warning: skipping unknown operator '{op}' in constraint ({a},{b},{op},{c}).")
            continue
        normalized_constraints.append((a,b,op,c))
    constraints = normalized_constraints

    # Compute feasible vertices
    vertices = _compute_vertices(constraints, nonneg)

    status, opt_val, opt_points = _classify(objective, sense, vertices, constraints, nonneg)

    if plot:
        _plot_lp(objective, sense, constraints, nonneg, vertices, status, opt_val, opt_points, show_objective_lines)

    # Reporting
    print("Status:", status)
    if status == 'optimal':
        print(f"Optimal value ({sense}) z = {opt_val:.4f} at {opt_points}")
    elif status == 'infinite':
        print(f"Infinite optimal solutions with objective value z = {opt_val:.4f} along segment {opt_points}")
    elif status == 'unbounded':
        print("Objective is unbounded in direction of optimization.")
    elif status == 'infeasible':
        print("No feasible region (infeasible LP).")

    return {
        'status': status,
        'optimal_value': opt_val,
        'optimal_points': opt_points,
        'vertices': vertices,
        'objective': objective,
        'sense': sense
    }


def _plot_lp(objective, sense, constraints, nonneg, vertices, status, opt_val, opt_points, show_objective_lines):
    c1, c2 = objective
    x_min, x_max, y_min, y_max = _generate_bounds(constraints, nonneg)
    xs = np.linspace(x_min, x_max, 400)
    ys = np.linspace(y_min, y_max, 400)
    X, Y = np.meshgrid(xs, ys)

    # Feasibility mask
    feasible_mask = np.ones_like(X, dtype=bool)
    for a,b,op,c in constraints:
        if op == '<=':
            feasible_mask &= (a*X + b*Y <= c + 1e-9)
        elif op == '>=':
            feasible_mask &= (a*X + b*Y >= c - 1e-9)
        elif op == '=':
            feasible_mask &= (np.abs(a*X + b*Y - c) <= 1e-3)  # equality creates thin region
    if nonneg:
        feasible_mask &= (X >= -1e-9) & (Y >= -1e-9)

    plt.figure(figsize=(8,6))
    plt.title("Graphical LP Solution (2 variables)")
    plt.xlabel("x")
    plt.ylabel("y")

    # Shade each constraint half-space lightly
    for a,b,op,c in constraints:
        if op == '<=':
            mask = (a*X + b*Y <= c + 1e-9)
        elif op == '>=':
            mask = (a*X + b*Y >= c - 1e-9)
        elif op == '=':
            mask = (np.abs(a*X + b*Y - c) <= 1e-3)
        else:
            continue
        plt.contourf(X, Y, mask, levels=[0,0.5,1], colors=[(1,1,1,0), (0.7,0.7,0.9,0.15)], alpha=0.15)

    # Feasible region shading
    if status != 'infeasible':
        plt.contourf(X, Y, feasible_mask, levels=[0,0.5,1], colors=[(1,1,1,0), (0.2,0.8,0.2,0.3)], alpha=0.3)

    # Plot constraint lines
    for a,b,op,c in constraints:
        line_y = None
        x_line_val = None
        if abs(b) > 1e-12:
            line_y = (c - a*xs)/b
            plt.plot(xs, line_y, 'k-', linewidth=1, zorder=3)
        else:
            # vertical line x = c/a
            if abs(a) > 1e-12:
                x_line_val = c / a
                plt.axvline(x_line_val, color='k', linewidth=1, zorder=3)
        # Label
        label = f"{a}x+{b}y{op}{c}"
        if line_y is not None:
            idx = np.argmin(np.abs(xs - (x_min + (x_max - x_min)*0.7)))
            y_label = line_y[idx]
            if np.isfinite(y_label) and y_min <= y_label <= y_max:
                plt.text(xs[idx], y_label, label, fontsize=8, color='black', zorder=5)
        elif x_line_val is not None and np.isfinite(x_line_val) and x_min <= x_line_val <= x_max:
            y_label = y_min + 0.7*(y_max - y_min)
            plt.text(x_line_val, y_label, label, fontsize=8, color='black', rotation=90, va='center', ha='left', zorder=5)

    if nonneg:
        plt.axvline(0, color='black', linewidth=1)
        plt.axhline(0, color='black', linewidth=1)

    # Mark vertices
    if vertices:
        vx, vy = zip(*vertices)
        plt.scatter(vx, vy, c='blue', s=40, label='Vertices')
        for (x,y) in vertices:
            plt.text(x, y, f"({x:.2f},{y:.2f})", fontsize=8, color='blue')

    # Objective lines
    if show_objective_lines and status in ('optimal','infinite') and opt_val is not None:
        # Choose several levels below optimum for max (or above for min)
        levels = []
        if sense == 'max':
            base = opt_val
            for k in [0.0, 0.2, 0.4, 0.6]:
                levels.append(base * (1 - k))
        else:
            base = opt_val
            for k in [0.0, 0.2, 0.4, 0.6]:
                levels.append(base * (1 + k))
        for z in levels:
            if abs(c2) > 1e-12:
                y_line = (z - c1*xs)/c2
                plt.plot(xs, y_line, '--', color='red', linewidth=1)
            else:
                x_line = z / c1
                plt.axvline(x_line, color='red', linestyle='--', linewidth=1)
        # Highlight optimal
        if opt_points:
            for (x,y) in opt_points:
                plt.scatter([x],[y], c='red', s=70, marker='*', label='Optimal')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# Example usage 
# ---------------------------------------------------------------
if __name__ == '__main__':
    # for function z = ax + by; objective = (a, b)
    # Sense: 'max' or 'min'
    # Subject to: (a, b, operator, constraint value) e.g. x <= 6 is (1, 0, '<=', 6)
    objective = (3, 2)
    sense = 'min'
    constraints = [ # (a, b, op, rhs)
        (1, 0, '>=', 0), # x >= 0
        (0, 1, '>=', 0), # y >= 0
        (1, 1, '<=', 5),
        (-1, 2, '<=', 4),
        (1, -1, '<=', 3),
        (2, 1, '>=', 12)
    ]
    solve_lp_2var(objective, sense, constraints, nonneg=False, plot=True)
    # after getting the optimal sol (x,y val), plug back into original objective to get optimal objective value
    # rmb as long as there is green region, there is feasible region
    # an unbounded feasible region does not mean LP is unbounded, depends on objective direction
