# Sample STRIPS Planner for Coffee Example

# Define the initial state
# Note: in our context, {} creates a SET (Python 3+ syntax), not a dictionary (no key-value pairs)
initial_state = {
    "Has(CoffeePot)",
    "On(Mug, Table)"
}

goal = {"Has(Coffee)"}

# STRIPS Action Definition
def action_pour(coffeepot, mug):
    """Pour coffee from pot into mug"""
    preconditions = {"Has(CoffeePot)", "On(Mug, Table)"}
    add_effects = {"Has(Coffee)"}
    del_effects = {"Has(CoffeePot)"}
    return (preconditions, add_effects, del_effects, f"Pour({coffeepot}, {mug})")

def is_goal_state(state, goal):
    """Check if the current state satisfies the goal"""
    return goal.issubset(state) # returns True if goal is a subset of current state.

def apply_action(state, action):
    """Apply an action to a state if preconditions are met"""
    preconditions, add_effects, del_effects, _ = action  # "_": Ignore action name string
    
    # Check if all preconditions are satisfied
    if not preconditions.issubset(state):
        return None  # Action cannot be applied
    
    # Apply the action effects
    new_state = state - del_effects
    new_state = new_state | add_effects # "|": set union operator. It combines two sets together.
    return new_state

def plan(initial_state, goal):
    """Simple BFS planner"""
    all_actions = [action_pour("CoffeePot", "Mug")]
    visited = set()  
    queue = []
    
    # Start with initial state and empty plan
    initial_tuple = tuple(sorted(initial_state))  # Convert to sorted tuple, so that it can be added into the visited set
    queue.append((initial_state, []))
    visited.add(initial_tuple)
    
    while queue:
        current_state, current_plan = queue.pop(0)
        
        if is_goal_state(current_state, goal):
            return current_plan
        
        for action in all_actions:
            new_state = apply_action(current_state, action)
            
            if new_state is None:
                continue
                
            # Convert state to sorted tuple for visited check
            new_state_tuple = tuple(sorted(new_state))
            
            if new_state_tuple not in visited:
                visited.add(new_state_tuple)
                preconditions, add_effects, del_effects, action_name = action
                queue.append((new_state, current_plan + [action_name]))
    
    return None

def main():
    """Main function to run the planner"""
    print("Initial State:")
    for fact in initial_state:
        print(f"  {fact}")
    
    print(f"\nGoal: {goal}")
    print("\nPlanning...")
    
    solution_plan = plan(initial_state, goal)
    
    if solution_plan:
        print("\nPlan found!")
        print("Sequence of actions:")
        for i, action in enumerate(solution_plan, 1):
            print(f"{i}. {action}")
    else:
        print("\nNo plan exists!")

# Entry point. Run main() only if this file is executed directly, not when imported
if __name__ == "__main__":
    main()  