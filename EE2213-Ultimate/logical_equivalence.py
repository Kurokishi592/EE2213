import numpy as np
from sympy import And, Not, satisfiable, Equivalent, Implies, Or, symbols
"""
Logical Equivalence Checking
Two sentences (formulas) alpha and beta are logically equivalent iff alpha ↔ beta is a tautology.
Equivalent tests:
  1. Tautology test: unsatisfiable(Not(Equivalent(alpha,beta)))
  2. Mutual entailment: alpha |= beta and beta |= alpha
  3. Truth-table: no valuation assigns different truth values.

Use logically_equivalent(alpha, beta) below. Set trace=True to list differing valuations (if any).
"""
def logically_equivalent(alpha, beta, method="auto", trace=False):
    """Return True iff alpha and beta are logically equivalent.

    method options:
      - 'auto' : use satisfiable-based tautology check (fast via SAT)
      - 'sat'  : same as auto
      - 'table': brute force over combined atoms
      - 'entail': mutual entailment using satisfiable()

    If trace=True and not equivalent, prints valuations where they differ.
    """
    atoms = sorted((alpha.atoms() | beta.atoms()), key=lambda s: str(s))
    # Fast SAT-based tautology check
    if method in ("auto", "sat"):
        eq = not satisfiable(And(alpha, Not(beta))) and not satisfiable(And(beta, Not(alpha)))
        if trace and not eq:
            _trace_diff(alpha, beta, atoms)
        return eq
    elif method == 'entail':
        eq = not satisfiable(And(alpha, Not(beta))) and not satisfiable(And(beta, Not(alpha)))
        if trace and not eq:
            _trace_diff(alpha, beta, atoms)
        return eq
    elif method == 'table':
        from itertools import product
        differing = []
        for vals in product([False, True], repeat=len(atoms)):
            env = {sym: val for sym, val in zip(atoms, vals)}
            aval = bool(alpha.subs(env))
            bval = bool(beta.subs(env))
            if aval != bval:
                differing.append(env)
        if trace and differing:
            print("[Equivalence Trace] Differing valuations:")
            for env in differing:
                env_str = ', '.join(f"{k}={env[k]}" for k in atoms)
                print(f"  {env_str} : {alpha}={bool(alpha.subs(env))}, {beta}={bool(beta.subs(env))}")
        return len(differing) == 0
    else:
        raise ValueError("Unsupported method for logical equivalence.")

def _trace_diff(alpha, beta, atoms):
    from itertools import product
    print("[Equivalence Trace] First few differing valuations (sat-based check failed mutual entailment):")
    shown = 0
    for vals in product([False, True], repeat=len(atoms)):
        env = {sym: val for sym, val in zip(atoms, vals)}
        aval = bool(alpha.subs(env))
        bval = bool(beta.subs(env))
        if aval != bval:
            env_str = ', '.join(f"{k}={env[k]}" for k in atoms)
            print(f"  {env_str} : {alpha}={aval}, {beta}={bval}")
            shown += 1
            if shown >= 5:
                break

# Example usage (uncomment to test):
# equivalence is A |= B and B |= A
A, B = symbols('A B')
alpha_example = And(A,B)
beta_example = Or(Not(A), B)  # logically equivalent to A -> B
print("Alpha:", alpha_example)
print("Beta :", beta_example)
print("Equivalent?", logically_equivalent(alpha_example, beta_example, method='auto', trace=True))