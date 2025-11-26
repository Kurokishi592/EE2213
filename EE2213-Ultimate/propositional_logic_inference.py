from sympy import symbols, And, Not, Or, Implies, Equivalent, satisfiable
import sys

'''
Knowledge Base Logic and Query Setup
'''
# Propositional symbols
A, B, C, D, E= symbols('A B C D E')  # A: Alex is a Knight, B: Ben is a Knight, C: Chloe is a Knight

# # Define the statements (Operator Precedence is important! () > ~ > & > | > Implies > Equivalent)
# if no brackets used e.g. P | Q & R should write Or(And(Q, R), P) to simulate the precedence
# another e.g. P∧ 𝑄 ⟺ ¬ 𝑄 ∨ R is Equivalent(Or(Not(B),C), And(A,B)) highest precedence is innermost
# XOR(A,B) is And(Or(A,B), Not(And(A,B)))

# statement_A = And(And(Implies(A, And(B,C)) , Or(A,C)) , Or(A,And(C,B)))
# statement_B = Implies(Not(A), And(Not(B), C))
# statement_C = Implies(And(Or(B,C), Not(And(B,C))), D)
# statement_D = Implies (D, A)

statement_A = B # A says B lies
statement_B = Not(C)  # B says A and B tell the truth
# statement_C

# statement_A = Equivalent(A, Implies(B, Not(C)))  # Alice says "if Bob tells the truth, then Carol lies"
# statement_B = Equivalent(B, And(Or(A,C), Not(And(A,C))))  # Bob says "Either Alice or Carol tells the truth (not both)"
# statement_C = Equivalent(C, Not(A))  # Carol says "Alice lies"

# Knowledge Base (KB)
KB = And(statement_A)#, statement_B, statement_C, statement_D)

query = A

''' If not running any specific, use this to do quick checks
Used for multiple queries at once or check entailment or equivalence A |= B AND B |= A '''
def Entails(KB_AND, query):
    entails = not satisfiable(And(KB_AND, Not(query)))
    if entails:
       print(query, "is definitely true.")
    else:
        entails_not = not satisfiable(And(KB_AND, query))
        if entails_not:
            print(query, "is definitely false.")
        else:
            print(query,"is uncertain.")

# This checks equivalence by mutual entailment
# Entails(Not(Or(A, And(B, Not(C)))), Or(And(Not(A), Not(B)), And(Not(A), C))) # can check entailment of left to right
# Entails(And(A,B), Equivalent(A, B))
# Entails(Implies(And(A,B), C), Or(Implies(A,C), Implies(B, C)))
# Entails(And(Or(A,B), Or(Or(Not(C),Not(D)),E)) , And(Or(A,B), Or(Not(D), E)) )

# this checks multiple queries at once
# Entails(KB, A)
# Entails(KB, B)
# Entails(KB, C)
# Entails(KB, D)

print("")

# --- Formatting utilities for symbolic logic output ---
def _supports_unicode(chars='¬∧∨→↔'):
    try:
        for ch in chars:
            ch.encode(sys.stdout.encoding or 'utf-8')
        return True
    except Exception:
        return False

USE_UNICODE = _supports_unicode()

def fmt(expr):
    """Format expression with unicode if console supports it, else ASCII."""
    NOT_SYM = '¬' if USE_UNICODE else '~'
    AND_SYM = ' ∧ ' if USE_UNICODE else ' & '
    OR_SYM = ' ∨ ' if USE_UNICODE else ' | '
    IMP_SYM = ' → ' if USE_UNICODE else ' -> '
    EQV_SYM = ' ↔ ' if USE_UNICODE else ' <-> '
    if expr.is_Symbol:
        return str(expr)
    if isinstance(expr, Not):
        return NOT_SYM + fmt(expr.args[0])
    if isinstance(expr, And):
        return '(' + AND_SYM.join(fmt(a) for a in expr.args) + ')'
    if isinstance(expr, Or):
        return '(' + OR_SYM.join(fmt(a) for a in expr.args) + ')'
    if isinstance(expr, Implies):
        return '(' + fmt(expr.args[0]) + IMP_SYM + fmt(expr.args[1]) + ')'
    if isinstance(expr, Equivalent):
        return '(' + EQV_SYM.join(fmt(a) for a in expr.args) + ')'
    return str(expr)

print("Knowledge Base (KB) statements:")
print("  1.", fmt(statement_A))
# print("  2.", fmt(statement_B))
# print("  3.", fmt(statement_C))
# print("  4.", fmt(statement_D))
print("KB (conjunction):", fmt(KB), "\n")

'''
Method 1: Model Checking Algorithm 
- Enumerate all possible models for (A, B, C) and check entailment of query using truth table. 
# KB |= query: for possible model, if KB is True, query is also True.
'''
from sympy.logic.boolalg import truth_table

# Generate truth table for KB
print("Truth table (models of A,B,C,D and KB evaluation):")
for assignment, Truth_value in truth_table(KB, [A, B, C]):
    a_val, b_val, c_val= assignment
    print(f"  A={a_val} B={b_val} C={c_val} | KB={Truth_value}") # C={c_val} D={d_val}

# KB |= query iff there is NO model where KB is True and query is False.
def entails_by_enumeration(alpha):
    """Return True iff KB |= alpha by brute-force enumeration (no model makes KB True and alpha False)."""
    for assignment, truth_val in truth_table(KB, [A, B, C]):
        if truth_val:  # KB true under this model
            # build env for query evaluation
            a_val, b_val, c_val = assignment
            env = {A: a_val, B: b_val, C: c_val}
            if not bool(alpha.subs(env)):
                return False
    return True

entails_q_enum = entails_by_enumeration(query)
entails_not_q_enum = entails_by_enumeration(Not(query))

if entails_q_enum and not entails_not_q_enum:
    enum_result = "Query entailed (True) via enumeration"
elif entails_not_q_enum and not entails_q_enum:
    enum_result = "Query negation entailed (False) via enumeration"
else:
    enum_result = "Query uncertain via enumeration"

print("\nEnumeration entailment result:")
print("  ", enum_result)

'''
Method 2: Refutation method: Satisfiable-based entailment check
KB |= query means there is no world where KB is true but query is false.
So it is same as checking KB ∧ ¬query is unsatisfiable (impossible)
'''
def entails_by_satisfiable(alpha):
    return not satisfiable(And(KB, Not(alpha)))

entails_q_sat = entails_by_satisfiable(query)
entails_not_q_sat = entails_by_satisfiable(Not(query))
if entails_q_sat and not entails_not_q_sat:
    sat_result = "Query entailed (True) via satisfiable()"
elif entails_not_q_sat and not entails_q_sat:
    sat_result = "Query negation entailed (False) via satisfiable()"
else:
    sat_result = "Query uncertain via satisfiable()"

print("\nSatisfiable-based entailment result:")
print("  ", sat_result)


'''
Method 3: Resolution Algorithm
- convert KB and Not(query) to CNF and Check if KB ⊨ 𝛼 Check if KB ∧ ¬𝛼 is contradiction by:
- Apply resolution rule to each pair of clauses that contains complementary literals to produce a new 
clause, which is added to the KB if it is not already present, until one of the following happens:
• No new clauses can be produced anymore : No entailment!
• Empty clause is produced (equivalent to False) : KB ⊨ 𝛼 
'''
from sympy.logic.boolalg import to_cnf

def _extract_clauses(expr):
    """Given a CNF sympy expression (And of Ors / literals) return list of frozenset literals."""
    if expr == True:  # tautology
        return []
    if isinstance(expr, Or):
        return [frozenset(_flatten_or(expr))]
    if isinstance(expr, And):
        clauses = []
        for arg in expr.args:
            if isinstance(arg, Or):
                clauses.append(frozenset(_flatten_or(arg)))
            else:
                clauses.append(frozenset([arg]))
        return clauses
    return [frozenset([expr])]

def _flatten_or(or_expr):
    stack = list(or_expr.args)
    lits = []
    while stack:
        item = stack.pop()
        if isinstance(item, Or):
            stack.extend(item.args)
        else:
            lits.append(item)
    return lits

def _complement(lit):
    return lit.args[0] if isinstance(lit, Not) else Not(lit)

def resolution_trace(expr, max_iterations=200):
    """Trace resolution on UNSIMPLIFIED CNF of expr (KB ∧ ¬query).
    Prints each resolvent. Returns True if empty clause derived, else False when saturated.
    """
    cnf_unsimplified = to_cnf(expr, simplify=False)
    print("\n[Resolution] Unsimplified CNF expression:")
    print("  ", cnf_unsimplified)
    # Extract initial clauses from unsimplified CNF (no simplify=True compression)
    clauses = set(_extract_clauses(cnf_unsimplified))
    print("[Resolution] Initial clauses:")
    for i, cl in enumerate(clauses, 1):
        print(f"  C{i}: {sorted(map(str, cl))}")
    iterations = 0
    clause_index = len(clauses)
    while iterations < max_iterations:
        iterations += 1
        print(f"\n[Resolution] Iteration {iterations}")
        new = set()
        cl_list = list(clauses)
        produced_any = False
        for i in range(len(cl_list)):
            for j in range(i+1, len(cl_list)):
                C1, C2 = cl_list[i], cl_list[j]
                # Look for complementary literals
                for lit in C1:
                    comp = _complement(lit)
                    if comp in C2:
                        resolvent_set = (C1 - {lit}) | (C2 - {comp})
                        # Skip tautological resolvents (contain L and ¬L)
                        if any(_complement(x) in resolvent_set for x in resolvent_set):
                            continue
                        fr = frozenset(resolvent_set)
                        print(f"    Resolve {sorted(map(str,C1))} & {sorted(map(str,C2))} on {str(lit)} -> {sorted(map(str,fr)) if fr else ['EMPTY']}")
                        if len(fr) == 0:
                            print("\n[Resolution] Empty clause derived => entailment holds.")
                            return True
                        if fr not in clauses and fr not in new:
                            new.add(fr)
                            produced_any = True
        if not produced_any:
            print("[Resolution] No new clauses ⇒ entailment fails.")
            return False
        added = 0
        for cl in new:
            if cl not in clauses:
                clause_index += 1
                print(f"    Added clause C{clause_index}: {sorted(map(str, cl))}")
                clauses.add(cl)
                added += 1
        if added == 0:
            print("[Resolution] Saturated without empty clause ⇒ entailment fails.")
            return False
    print("[Resolution] Iteration cap reached without empty clause ⇒ entailment fails.")
    return False

# print("\n=== Resolution Trace ===")
# res_success = resolution_trace(And(KB, Not(query)))
# res_result = "Query entailed (True) via resolution trace" if res_success else "Query not entailed via resolution trace"
# print("\nResolution trace result:")
# print("  ", res_result)

print("\nAll methods comparison:")
print(f"  Enumeration : {enum_result}")
print(f"  Satisfiable : {sat_result}")
# print(f"  Resolution  : {res_result}")

