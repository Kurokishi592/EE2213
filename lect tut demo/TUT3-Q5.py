# Define the initial state
# Note: in our context, {} creates a SET (Python 3+ syntax), not a dictionary (no key-value pairs)
initial_state = {
    "Has(CoffeePowder)",
    "Has(TapWater)",          # Water from tap - unlimited
    "On(CoffeePot, Table)",
    "On(Mug, Table)",
    "Has(MilkInMilkJug)"        
}

goal = {"Has(CoffeeWithMilkInMug)"}

# STRIPS Action Definition
def action_pour_water_into_pot():
    """Pour tap water into the coffee pot"""
    preconditions = {"On(CoffeePot, Table)", "Has(TapWater)"}
    add_effects = {"Has(ColdWaterInPot)"}
    del_effects = set()
    return (preconditions, add_effects, del_effects, "PourWaterIntoPot()")

def action_boil_water_in_pot():
    """Boil the water in the pot"""
    preconditions = {"Has(ColdWaterInPot)"}
    add_effects = {"Has(BoiledWaterInPot)"}
    del_effects = {"Has(ColdWaterInPot)"}
    return (preconditions, add_effects, del_effects, "BoilWaterInPot()")

def action_make_coffee():
    """Make coffee by combining coffee powder and boiled water"""
    preconditions = {"Has(CoffeePowder)", "Has(BoiledWaterInPot)"}
    add_effects = {"Has(CoffeeInPot)"}
    del_effects = {"Has(CoffeePowder)", "Has(BoiledWaterInPot)"}
    return (preconditions, add_effects, del_effects, "MakeCoffee()")

def action_pour_coffee_into_mug():
    """Pour coffee from pot into mug"""
    preconditions = {"Has(CoffeeInPot)", "On(Mug, Table)"}
    add_effects = {"Has(CoffeeInMug)"}
    del_effects = {"Has(CoffeeInPot)"}
    return (preconditions, add_effects, del_effects, "PourCoffeeIntoMug()")

def action_add_milk():
    """Add milk to coffee in mug"""
    preconditions = {"Has(CoffeeInMug)", "Has(MilkInMilkJug)"}
    add_effects = {"Has(CoffeeWithMilkInMug)"}
    del_effects = {"Has(MilkInMilkJug)"}  # Suppose milk is used up
    return (preconditions, add_effects, del_effects, "AddMilk()")

def is_goal_state(state, goal):
    return goal.issubset(state)  # returns True if goal is a subset of current state.

def apply_action(state, action):
    preconditions, add_effects, del_effects, _ = action  # "_": Ignore action name string

    # Check if all preconditions are satisfied    
    if not preconditions.issubset(state):
        return None # Action cannot be applied
    
    # Apply the action effects
    new_state = (state - del_effects) | add_effects
    return new_state

def plan(initial_state, goal):
    """BFS planner"""
    all_actions = [
        action_pour_water_into_pot(),
        action_boil_water_in_pot(),
        action_make_coffee(),
        action_pour_coffee_into_mug(),
        action_add_milk()
    ]
    visited = set()
    queue = []

    # Start with initial state and empty plan 
    initial_tuple = tuple(sorted(initial_state))
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
                _, _, _, action_name = action
                queue.append((new_state, current_plan + [action_name]))

    return None

def main():
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
