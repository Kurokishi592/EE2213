def ProjectedGradientDescent(f, f_prime=None, initial=None, learning_rate=0.1, num_iters=100,
                             round_output=True, decimals=4, projection=None):
    """Projected Gradient Descent for TWO OR MORE variables (compatible interface).

    Mirrors the original GradientDescent function signature and behavior, with one
    additional optional argument:
        projection: callable taking a point x and returning its projection onto
                    the feasible set. If None, behaves exactly like plain gradient
                    descent. The projection is applied AFTER the gradient step.

    Returns (steps, fvals, grads, projections)
        steps       : sequence of iterates (including initial)
        fvals       : objective values at each iterate
        grads       : gradient at each iterate
        projections : sequence of projected points AFTER projection (same length as steps);
                      if projection is None, this is identical to steps.

    Projection function examples:
        1) Box constraints: lambda x: np.clip(x, lower_bounds, upper_bounds)
        2) Non-negativity: lambda x: np.maximum(x, 0)
        3) L2 ball of radius R: lambda x: x * min(1, R / np.linalg.norm(x))

    The rest of the behavior (symbolic gradient attempt, finite-difference fallback,
    scalar handling, rounding) matches GradientDescent.
    """
    import numpy as np

    if initial is None:
        raise ValueError("initial must be provided")

    original_arr = np.asarray(initial)
    scalar_input = (original_arr.ndim == 0)
    x0 = np.atleast_1d(np.asarray(initial, dtype=float))
    steps = np.array([x0])
    proj_steps = np.array([x0 if projection is None else np.atleast_1d(projection(x0))])

    numeric_grad = None
    if f_prime is None:
        try:
            import sympy as sp
            import types
            n = x0.size
            syms = sp.symbols('x0:%d' % n)
            if callable(f):
                try:
                    new_globals = dict(getattr(f, '__globals__', {}))
                    new_globals.setdefault('np', sp)
                    new_globals.setdefault('sympy', sp)
                    new_globals.setdefault('math', sp)
                    newf = types.FunctionType(f.__code__, new_globals)
                    expr = newf(syms if n > 1 else syms[0])
                except Exception:
                    expr = f(syms if n > 1 else syms[0])
            else:
                expr = sp.sympify(f)
            expr = sp.simplify(expr)
            if n == 1:
                derivs = [sp.diff(expr, syms[0])]
            else:
                derivs = [sp.diff(expr, s) for s in syms]
            lambdified = [sp.lambdify(syms, d, modules=['numpy']) for d in derivs]
            def numeric_grad(x):
                x = np.asarray(x, dtype=float)
                try:
                    vals = [lf(*x) if isinstance(x, (list, tuple, np.ndarray)) else lf(x) for lf in lambdified]
                except Exception:
                    vals = [lf(x) for lf in lambdified]
                vals = np.asarray(vals, dtype=float)
                return vals if vals.size > 1 else np.asarray([float(vals)])
        except Exception:
            numeric_grad = None

    def get_grad(x):
        if f_prime is not None:
            val = f_prime(x if not (scalar_input and x.size == 1) else x[0])
            return np.atleast_1d(np.asarray(val, dtype=float))
        if numeric_grad is not None:
            return np.atleast_1d(np.asarray(numeric_grad(x), dtype=float))
        eps = 1e-6
        x = np.asarray(x, dtype=float)
        grad = np.zeros_like(x, dtype=float)
        for i in range(x.size):
            x_eps = x.copy()
            x_eps[i] += eps
            f_plus = float(f(x_eps if not (scalar_input and x_eps.size == 1) else x_eps[0]))
            x_eps[i] -= 2 * eps
            f_minus = float(f(x_eps if not (scalar_input and x_eps.size == 1) else x_eps[0]))
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    for iteration in range(num_iters):
        current = steps[iteration]
        grad = get_grad(current)
        tentative = current - learning_rate * grad
        projected = tentative if projection is None else np.atleast_1d(projection(tentative))
        steps = np.vstack((steps, projected))
        proj_steps = np.vstack((proj_steps, projected))

    fvals = np.array([float(f(s if not (scalar_input and s.size == 1) else s[0])) for s in steps], dtype=float)
    grads = np.array([get_grad(s) for s in steps], dtype=float)

    if scalar_input:
        steps = steps.flatten()
        grads = grads.flatten()
        proj_steps = proj_steps.flatten()

    if round_output:
        steps = np.round(steps, decimals)
        fvals = np.round(fvals, decimals)
        grads = np.round(grads, decimals)
        proj_steps = np.round(proj_steps, decimals)

    return steps, fvals, grads, proj_steps

def make_halfspace_projector(a, b):
    """Return a projector onto the closed half-space {x | a^T x <= b}.

    Parameters:
        a : 1-D iterable (normal vector)
        b : scalar (offset)

    Projection formula for a violated point y (a^T y > b):
        y_proj = y - (a^T y - b) / ||a||^2 * a
    If already feasible, y is returned unchanged.
    """
    import numpy as np
    a = np.atleast_1d(np.asarray(a, dtype=float))
    a_norm_sq = float(np.dot(a, a))
    if a_norm_sq == 0:
        raise ValueError("Normal vector 'a' must be non-zero")
    def projector(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        val = float(np.dot(a, x))
        if val <= b:
            return x
        return x - (val - b) / a_norm_sq * a
    return projector

def make_hyperplane_projector(a, b):
    """Return a projector onto the hyperplane {x | a^T x = b}. Always adjusts.

    Projection formula: y_proj = y - (a^T y - b) / ||a||^2 * a
    """
    import numpy as np
    a = np.atleast_1d(np.asarray(a, dtype=float))
    a_norm_sq = float(np.dot(a, a))
    if a_norm_sq == 0:
        raise ValueError("Normal vector 'a' must be non-zero")
    def projector(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return x - (float(np.dot(a, x)) - b) / a_norm_sq * a
    return projector

def make_polyhedron_projector(A, b, passes=3):
    """Return a simple sequential projector for Ax <= b.

    Parameters:
        A : 2-D iterable (m x n) constraint matrix
        b : 1-D iterable (m,) offsets
        passes : number of sweeps over all constraints (POCS style)

    Method: For each violated inequality a_i^T x > b_i, apply the half-space
    projection. Repeat for 'passes' sweeps (not guaranteed exact for coupled
    active sets but often sufficient for simple uses).
    """
    import numpy as np
    A = np.asarray(A, dtype=float)
    b = np.atleast_1d(np.asarray(b, dtype=float))
    if A.ndim != 2:
        raise ValueError("A must be a 2-D array")
    if b.shape[0] != A.shape[0]:
        raise ValueError("b length must match number of rows in A")
    norms_sq = np.sum(A * A, axis=1)
    if np.any(norms_sq == 0):
        raise ValueError("Each row of A must be non-zero")
    def projector(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = x.copy()
        for _ in range(passes):
            vals = A @ y - b
            violated = vals > 0
            if not np.any(violated):
                break
            for i in np.where(violated)[0]:
                y = y - vals[i] / norms_sq[i] * A[i]
        return y
    return projector

def SimpleProjectedGradientDescent(f, constraints, initial, learning_rate=0.1, num_iters=100,
                                   round_output=True, decimals=4, verbose=True):
    """Simplified Projected Gradient Descent.

    Inputs (mirrors GradientDescent but adds constraints):
        f            : callable or sympy‑compatible expression of variables (vector argument)
          constraints  : accepted formats (all converted to a_i^T x <= b_i):
                              1) [(a_vec, b_scalar), ...] with a_vec length = dimension (direct a^T x <= b)
                              2) Legacy 2D: [(a1, a2, 'op', rhs), ...] treated as a1*x + a2*y op rhs
                              3) General multi-var: [([a1,a2,...,ak], 'op', rhs), ...]
                              4) 1D legacy: [(a, 0, 'op', rhs), ...] second value must be 0
                              5) 1D compact: [(a, 'op', rhs), ...]
                                  op in {'<=','>=','=','=<','=>'}; '=' expands into two inequalities.
        initial      : iterable initial point
        learning_rate: step size
        num_iters    : number of iterations

    Automatic gradient: attempts symbolic gradient via sympy; falls back to finite differences.

    Projection set: Intersection of all half-spaces a_i^T x <= b_i (convex polyhedron).
    Projection method: Sequential projections (POCS style) using make_polyhedron_projector.

    Printed each iteration (if verbose):
        iter, x_before, grad, tentative, x_projected, each a_i^T x_projected, sum_of_products = sum_i a_i^T x_projected

    Returns: (steps, fvals, grads, proj_steps, constraint_dot_products)
        steps: points after projection (including initial projected)
        fvals: objective values at each recorded step
        grads: gradient at each step (symbolic/numeric)
        proj_steps: alias of steps (for interface similarity)
        constraint_dot_products: list of lists, each inner list has a_i^T x for all constraints at that step
    """
    import numpy as np
    if initial is None:
        raise ValueError("initial must be provided")
    # Build projector from any accepted constraint syntax
    A_list = []
    b_list = []
    dim = len(initial) if not isinstance(initial, (int,float)) else 1
    if not constraints:
        raise ValueError("At least one constraint required (use a very large box if unconstrained).")

    first_len = len(constraints[0])
    if first_len == 2:
        # Format 1: (a_vec, b)
        for (a_vec, b_scalar) in constraints:
            a_arr = np.atleast_1d(np.asarray(a_vec, dtype=float))
            if a_arr.ndim != 1:
                raise ValueError("Constraint normal must be 1-D")
            if a_arr.size != dim:
                raise ValueError(f"Constraint normal dimension {a_arr.size} != variable dimension {dim}")
            A_list.append(a_arr)
            b_list.append(float(b_scalar))
    elif first_len == 4 and not isinstance(constraints[0][0], (list, tuple)):  # legacy (a,b,op,rhs)
        if dim == 1:
            # Treat (a, b, op, rhs) with b expected 0; ignore b
            for a, b, op, rhs in constraints:
                if abs(b) > 1e-12:
                    raise ValueError("For 1D legacy syntax (a,b,op,rhs), b must be 0.")
                if op == '=>': op = '>='
                elif op == '=<': op = '<='
                if op not in ('<=','>=','='):
                    raise ValueError(f"Unknown operator '{op}' in constraint {(a,b,op,rhs)}")
                a_vec = np.asarray([a], dtype=float)
                if op == '<=':
                    A_list.append(a_vec); b_list.append(float(rhs))
                elif op == '>=':
                    A_list.append(-a_vec); b_list.append(float(-rhs))
                else:  # '='
                    A_list.append(a_vec); b_list.append(float(rhs))
                    A_list.append(-a_vec); b_list.append(float(-rhs))
        elif dim == 2:
            for a, b, op, rhs in constraints:
                if op == '=>': op = '>='
                elif op == '=<': op = '<='
                if op not in ('<=','>=','='):
                    raise ValueError(f"Unknown operator '{op}' in constraint {(a,b,op,rhs)}")
                a_vec = np.asarray([a, b], dtype=float)
                if op == '<=':
                    A_list.append(a_vec); b_list.append(float(rhs))
                elif op == '>=':
                    A_list.append(-a_vec); b_list.append(float(-rhs))
                else:  # '='
                    A_list.append(a_vec); b_list.append(float(rhs))
                    A_list.append(-a_vec); b_list.append(float(-rhs))
        else:
            raise ValueError("Legacy (a,b,op,rhs) syntax only valid for 1 or 2 variables; use ([...],op,rhs) for higher dimensions.")
    elif first_len == 3 and not isinstance(constraints[0][0], (list, tuple)) and dim == 1:
        # Single variable compact (a, op, rhs)
        for a, op, rhs in constraints:
            if op == '=>': op = '>='
            elif op == '=<': op = '<='
            if op not in ('<=','>=','='):
                raise ValueError(f"Unknown operator '{op}' in constraint {(a,op,rhs)}")
            a_vec = np.asarray([a], dtype=float)
            if op == '<=':
                A_list.append(a_vec); b_list.append(float(rhs))
            elif op == '>=':
                A_list.append(-a_vec); b_list.append(float(-rhs))
            else:  # '='
                A_list.append(a_vec); b_list.append(float(rhs))
                A_list.append(-a_vec); b_list.append(float(-rhs))
    elif first_len == 3 and isinstance(constraints[0][0], (list, tuple)):
        # Format 3: ([coeffs], op, rhs)
        for coeffs, op, rhs in constraints:
            if op == '=>': op = '>='
            elif op == '=<': op = '<='
            if op not in ('<=','>=','='):
                raise ValueError(f"Unknown operator '{op}' in constraint {(coeffs,op,rhs)}")
            a_vec = np.atleast_1d(np.asarray(coeffs, dtype=float))
            if a_vec.ndim != 1:
                raise ValueError("Coefficient vector must be 1-D")
            if a_vec.size != dim:
                raise ValueError(f"Constraint coefficient length {a_vec.size} != variable dimension {dim}")
            if op == '<=':
                A_list.append(a_vec); b_list.append(float(rhs))
            elif op == '>=':
                A_list.append(-a_vec); b_list.append(float(-rhs))
            else:  # '='
                A_list.append(a_vec); b_list.append(float(rhs))
                A_list.append(-a_vec); b_list.append(float(-rhs))
    else:
        raise ValueError("Unsupported constraint format. Use (a_vec,b), (a,b,op,rhs) for 2D, or ([coeffs],op,rhs) for general dimension.")

    A = np.vstack(A_list)
    b_vec = np.asarray(b_list)
    projector = make_polyhedron_projector(A, b_vec, passes=3)

    original_arr = np.asarray(initial)
    scalar_input = (original_arr.ndim == 0)
    x0 = np.atleast_1d(np.asarray(initial, dtype=float))
    # Keep raw initial (may be infeasible) per revised iteration semantics
    steps = np.array([x0])
    constraint_values = []  # record only after projection each iteration

    # Attempt symbolic gradient
    numeric_grad = None
    try:
        import sympy as sp, types
        n = x0.size
        syms = sp.symbols('x0:%d' % n)
        if callable(f):
            try:
                new_globals = dict(getattr(f, '__globals__', {}))
                new_globals.setdefault('np', sp)
                new_globals.setdefault('sympy', sp)
                new_globals.setdefault('math', sp)
                newf = types.FunctionType(f.__code__, new_globals)
                expr = newf(syms if n > 1 else syms[0])
            except Exception:
                expr = f(syms if n > 1 else syms[0])
        else:
            expr = sp.sympify(f)
        expr = sp.simplify(expr)
        derivs = [sp.diff(expr, s) for s in syms]
        lambdified = [sp.lambdify(syms, d, modules=['numpy']) for d in derivs]
        def numeric_grad(x):
            x = np.asarray(x, dtype=float)
            try:
                vals = [lf(*x) if isinstance(x, (list, tuple, np.ndarray)) else lf(x) for lf in lambdified]
            except Exception:
                vals = [lf(x) for lf in lambdified]
            vals = np.asarray(vals, dtype=float)
            return vals if vals.size > 1 else np.asarray([float(vals)])
    except Exception:
        numeric_grad = None

    def get_grad(x):
        if numeric_grad is not None:
            return np.atleast_1d(np.asarray(numeric_grad(x), dtype=float))
        # finite differences
        eps = 1e-6
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        for i in range(x.size):
            x_eps = x.copy(); x_eps[i] += eps
            f_plus = float(f(x_eps if not (scalar_input and x_eps.size == 1) else x_eps[0]))
            x_eps[i] -= 2*eps
            f_minus = float(f(x_eps if not (scalar_input and x_eps.size == 1) else x_eps[0]))
            g[i] = (f_plus - f_minus)/(2*eps)
        return g

    if verbose:
        print("Projected Gradient Descent (linear constraints)\n")

    for it in range(num_iters):
        # First iteration uses raw initial x0 (possibly infeasible)
        x_curr = steps[it]
        grad = get_grad(x_curr)
        tentative = x_curr - learning_rate * grad
        x_proj = projector(tentative)
        steps = np.vstack((steps, x_proj))
        cv = A @ x_proj
        constraint_values.append(list(cv))
        if verbose:
            print(f"Iter {it+1}:")
            print(f"  x_before     = {x_curr}")
            print(f"  grad         = {grad}")
            print(f"  tentative    = {tentative}")
            print(f"  x_projected  = {x_proj}")
            print(f"  a_i^T x_proj = {cv}  Sum: {cv.sum()}")
            print("----")

    fvals = np.array([float(f(s if not (scalar_input and s.size == 1) else s[0])) for s in steps], dtype=float)
    grads = np.array([get_grad(s) for s in steps], dtype=float)
    proj_steps = steps.copy()

    if scalar_input:
        steps = steps.flatten(); grads = grads.flatten(); proj_steps = proj_steps.flatten()
    if round_output:
        steps = np.round(steps, decimals); fvals = np.round(fvals, decimals); grads = np.round(grads, decimals); proj_steps = np.round(proj_steps, decimals)
    return steps, fvals, grads, proj_steps, constraint_values

if __name__ == "__main__":
    import numpy as np
    # Example: minimize f(x,y)=x^2 + y^2 subject to x,y >= 0 and x + y <= 1
    def f(v):
        x,y = v
        return x**2 + y**2
    def f_prime(v):
        x,y = v
        return np.array([2*x, 2*y])
    def projection(v):
        x,y = v
        # project onto nonnegativity
        x = max(0,x); y = max(0,y)
        # if outside x+y<=1, project onto line x+y=1 in L2 sense
        if x + y > 1:
            s = x + y
            x = x / s
            y = y / s
        return np.array([x,y])
    steps, fvals, grads, proj = ProjectedGradientDescent(f, f_prime=f_prime, initial=[0.8,0.8], learning_rate=0.2, num_iters=20, projection=projection)
    print("Steps:\n", steps)
    print("Projected Steps:\n", proj)
    print("Values:\n", fvals)
    print("Grads:\n", grads)

    # Additional demo: quadratic minimization with a single linear half-space constraint
    # Minimize g(x,y) = (x-2)^2 + (y+1)^2 subject to x + y <= 1
    def g(v):
        x,y = v
        return (x-2)**2 + (y+1)**2
    def g_prime(v):
        x,y = v
        return np.array([2*(x-2), 2*(y+1)])
    halfspace_proj = make_halfspace_projector([1,1], 1)
    g_steps, g_vals, g_grads, g_proj = ProjectedGradientDescent(g, f_prime=g_prime, initial=[3.0,0.0], learning_rate=0.1, num_iters=25, projection=halfspace_proj)
    print("\nHalf-space constrained descent (x + y <= 1):")
    print("Steps:\n", g_steps)
    print("Objective values:\n", g_vals)

    # Simple interface demo: minimize h(x,y)=(x-1)^2+(y-2)^2 with constraints x>=0, y>=0, x+y<=2
    def h(v):
        x,y = v
        return (x-1)**2 + (y-2)**2
    # Demo with LP-style syntax: x >=0, y >=0, x + y <= 2
    constraints_lp = [
        (1, 0, '>=', 0),
        (0, 1, '>=', 0),
        (1, 1, '<=', 2)
    ]
    print("\nSimpleProjectedGradientDescent demo (LP-style syntax):\n")
    sp_steps, sp_vals, sp_grads, sp_proj, sp_cvals = SimpleProjectedGradientDescent(h, constraints_lp, initial=[3.0,-1.0], learning_rate=0.15, num_iters=5)
    print("Final projected steps:\n", sp_steps)
    print("Objective values:\n", sp_vals)
    print("Constraint dot products per step:\n", sp_cvals)
    # 3-variable demo: minimize (x-1)^2 + (y-2)^2 + (z+1)^2 subject to x,y,z>=0 and x+y+z <= 5
    def h3(v):
        x,y,z = v
        return (x-1)**2 + (y-2)**2 + (z+1)**2
    constraints_3d = [
        ([1,0,0], '>=', 0),
        ([0,1,0], '>=', 0),
        ([0,0,1], '>=', 0),
        ([1,1,1], '<=', 5)
    ]
    print("\nSimpleProjectedGradientDescent demo (3D constraints):\n")
    sp3_steps, sp3_vals, sp3_grads, sp3_proj, sp3_cvals = SimpleProjectedGradientDescent(h3, constraints_3d, initial=[2.5,-1.0,4.0], learning_rate=0.12, num_iters=5)
    print("Final projected 3D steps:\n", sp3_steps)
    print("Objective values 3D:\n", sp3_vals)
    print("Constraint dot products per 3D step:\n", sp3_cvals)