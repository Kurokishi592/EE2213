def GradientDescent(f, f_prime=None, initial=None, learning_rate=0.1, num_iters=100, round_output=True, decimals=4):
    import numpy as np

    # prepare initial point
    if initial is None:
        raise ValueError("initial must be provided")

    # remember if user passed a scalar initial (so we can keep output shapes identical)
    original_arr = np.asarray(initial)
    scalar_input = (original_arr.ndim == 0)

    x0 = np.atleast_1d(np.asarray(initial, dtype=float))
    steps = np.array([x0])

    # if no analytic gradient provided, try to construct one using sympy
    numeric_grad = None
    if f_prime is None:
        try:
            import sympy as sp
            import types

            n = x0.size
            # create sympy symbols x0, x1, ...
            syms = sp.symbols('x0:%d' % n)

            # attempt to call f on sympy symbols. If f references numpy (np), recreate a function
            # that maps 'np' to sympy so expressions like np.cos translate to sympy.cos
            if callable(f):
                try:
                    # recreate function with modified globals mapping np->sympy, math->sympy where present
                    new_globals = dict(getattr(f, '__globals__', {}))
                    # map common modules to sympy
                    new_globals.setdefault('np', sp)
                    new_globals.setdefault('sympy', sp)
                    new_globals.setdefault('math', sp)
                    newf = types.FunctionType(f.__code__, new_globals)
                    expr = newf(syms if n > 1 else syms[0])
                except Exception:
                    # fallback: try calling original f directly (may work if it uses plain python ops)
                    expr = f(syms if n > 1 else syms[0])
            else:
                # if f is a string or sympy expression, sympify it
                expr = sp.sympify(f)

            # ensure expr is a sympy expression
            expr = sp.simplify(expr)

            # compute partial derivatives
            if n == 1:
                derivs = [sp.diff(expr, syms[0])]
            else:
                derivs = [sp.diff(expr, s) for s in syms]

            # convert sympy derivatives to numeric functions using lambdify
            lambdified = [sp.lambdify(syms, d, modules=['numpy']) for d in derivs]

            def numeric_grad(x):
                x = np.asarray(x, dtype=float)
                # lambdified expects separate args, so unpack
                try:
                    vals = [lf(*x) if isinstance(x, (list, tuple, np.ndarray)) else lf(x) for lf in lambdified]
                except Exception:
                    vals = [lf(x) for lf in lambdified]
                vals = np.asarray(vals, dtype=float)
                # if original input was scalar and n==1, return scalar-like array
                return vals if vals.size > 1 else np.asarray([float(vals)])

        except Exception:
            # If sympy is not available or symbolic conversion fails, fall back to numerical gradient
            numeric_grad = None

    # if f_prime provided by caller, use it; else use numeric_grad if available; otherwise use finite differences
    def get_grad(x):
        # x is a 1-D array internally. If caller-provided f_prime expects scalar, pass scalar when appropriate
        if f_prime is not None:
            val = f_prime(x if not (scalar_input and x.size == 1) else x[0])
            return np.atleast_1d(np.asarray(val, dtype=float))
        if numeric_grad is not None:
            return np.atleast_1d(np.asarray(numeric_grad(x), dtype=float))

        # finite difference approximation
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
        grad = get_grad(steps[iteration])
        new_step = steps[iteration] - learning_rate * grad
        steps = np.vstack((steps, new_step))

    # when evaluating f and gradients, pass scalar to f if user originally provided scalar
    fvals = np.array([float(f(s if not (scalar_input and s.size == 1) else s[0])) for s in steps], dtype=float)
    grads = np.array([get_grad(s) for s in steps], dtype=float)

    # if user passed scalar initial, keep output shapes compatible with old behavior
    if scalar_input:
        steps = steps.flatten()
        grads = grads.flatten()

    if round_output:
        steps = np.round(steps, decimals)
        fvals = np.round(fvals, decimals)
        grads = np.round(grads, decimals)

    return steps, fvals, grads