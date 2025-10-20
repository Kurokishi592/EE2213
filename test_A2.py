import A2_A0303203A as grading

road_map = {
    'A': {'B': 4, 'C': 6, 'D': 3},
    'B': {'F': 6},
    'C': {'D': 2, 'E': 3, 'F': 3},
    'D': {'E': 4},
    'E': {'F': 1},
    'F': {}
}

city_coordinates = {
    'A': (1, 2),
    'B': (2, 5),
    'C': (5, 2),
    'D': (7, 4),
    'E': (9, 5),
    'F': (10, 8)
}

# Run the A* algorithm
path, cost = grading.A2_A0303203A(road_map, city_coordinates, 'F', 'A')
print("Shortest path:", path)
print("Total distance:", cost)
