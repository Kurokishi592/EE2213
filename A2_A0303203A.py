import numpy as np
import heapq

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0303203A(road_map, city_coordinates, start_city, destination_city):
    """
    Input types:
    :road_map type: dict
        Dictionary representing the weighted graph. 
        Example: {'A': {'B': 5, 'C': 10}, 'B': {'D': 2}, ...}

    :city_coordinates type: dict
        Dictionary mapping city names to (x, y) coordinates. 
        Example: {'A': (0, 0), 'B': (3, 4), 'C': (6, 0), ...}

    :start_city type: str
    :destination_city type: str

    Return types:
    :shortest_path type: list
        List of city names from start to destination (inclusive).

    :total_cost type: float
        Total path cost (sum of weights along the path).
    """

    # Heuristic function: Euclidean distance between two cities
    def heuristic(city1, city2):
        coord1 = np.array(city_coordinates[city1])
        coord2 = np.array(city_coordinates[city2])
        return np.linalg.norm(coord1 - coord2)
      

    # Hint 1: You may want to use a dictionary to store the best known cost (g-score) and path for each city.
    # Example structure (you may use a different structure if you prefer): node_data = {city: {'dist': ..., 'prev': [...]}}

    # Hint 2: You will need a priority queue (e.g., min-heap) to explore cities in order of their f = g + h values.
    # You can use heapq.heappush() and heapq.heappop() for priority queue operations.

    # Initialize the data structures here
    min_heap = []
    heapq.heappush(min_heap, (0 + heuristic(start_city, destination_city), start_city))

    # Initialize the g_score (actual cost) 
    g_score = {city: np.inf for city in road_map}
    g_score[start_city] = 0
    
    # For reconstructing the path
    came_from = {}
    
    # Main A* search loop
    while min_heap:
        current_city = heapq.heappop(min_heap)[1]
        if current_city == destination_city:
            # Reconstruct the path
            path = [current_city]
            while current_city in came_from:
                current_city = came_from[current_city]
                path.append(current_city)
            path.reverse()
            return path, float(g_score[destination_city])

        # Explore neighbors
        for neighbor, cost in road_map[current_city].items():
            tentative_g_score = g_score[current_city] + cost
            # Relaxation step
            if tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, destination_city)
                heapq.heappush(min_heap, (f_score, neighbor))
                came_from[neighbor] = current_city

    # If no path is found, return None and np.inf
    # (Do not change or modify this line.)
    return None, np.inf  
