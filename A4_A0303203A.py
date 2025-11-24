from sympy import symbols, And, Not, Or, Implies, Equivalent
from sympy.logic.boolalg import truth_table

# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A4_A0303203A(query):
    """
    Args:
        query: A sympy logical expression representing the query to be checked
               (e.g., A, Not(B), etc.).

    Returns:
        result: A string "True" if the query is a Knight;
                A string "False" if the query is a Knave;
                A string "Not Sure" if the type of the query cannot be determined.
    """

    # Propositional symbols
    A, B, C = symbols('A B C')  # A: Alex is a Knight, B: Ben is a Knight, C: Chloe is a Knight
    
    statement_A = Equivalent(A, Not(B)) #"B is a Knave" 
    statement_B = Equivalent(B, Not(Equivalent(A, B))) #"A and B are of different types" (XOR)
    statement_C = Equivalent(C, Or(A, C)) #"At least one of A and C is a Knight"

    # Knowledge Base (KB)
    KB = And(statement_A, statement_B, statement_C)

    # Model-checking algorithm:
    # KB |= alpha if KB is True, alpha is also True, using truth_table to check entailment for EVERY model.
    # This means KB |= alpha iff there is NO model where KB is True and alpha is False.
    def entails(alpha):
        for assignment, Truth_val in truth_table(And(KB, Not(alpha)), [A, B, C]):
            if Truth_val:
                # found a model where KB is True and alpha is False => not entailed
                return False
        return True

    entails_q = entails(query) # this is to check if query is true, but we also need to know if query is false
    entails_not_q = entails(Not(query)) # hence we check for Not(query) as well

    # Decide result
    if entails_q and not entails_not_q:
        result = "True"
    elif entails_not_q and not entails_q:
        result = "False"
    else:
        result = "Not Sure"

    # return in this order
    return result
