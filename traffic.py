"""
Traffic Simulation - Base Version for Issue Assignments
This code works but has known issues for participants to fix/improve

EFFICIENCY ANALYSIS & LIBRARY ALTERNATIVES:

Current Implementation Analysis:
- Uses heapq (binary heap) for priority queue operations
- Time Complexity: O(E log V) where E = edges, V = vertices
- Space Complexity: O(V + E) for graph storage and distances
- Zero external dependencies, highly transparent logic

Library Comparison:
┌─────────────────┬──────────────────────────┬────────────────────────────────┐
│ Library         │ Key Benefits             │ Trade-offs / When to Use       │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ stdlib + heapq  │ Zero dependencies,       │ Manual path reconstruction,    │
│ (CURRENT)       │ O(E log V) performance,  │ slower than C++ backends for   │
│                 │ transparent logic        │ massive graphs (V,E >> 10^5)  │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ NetworkX        │ Ease of use, wide array  │ Pure Python overhead, slower   │
│                 │ of algorithms, prototyping│ for performance-critical tasks │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ graph-tool      │ C++/Boost backend,       │ Complex dependencies, steep    │
│                 │ massive graph performance │ learning curve, compilation    │
├─────────────────┼──────────────────────────┼────────────────────────────────┤
│ igraph          │ Fast C core, language    │ External dependency, best for  │
│                 │ agnostic performance     │ high performance needs         │
└─────────────────┴──────────────────────────┴────────────────────────────────┘

RECOMMENDATION: Current implementation is optimal for typical traffic networks.
Libraries like graph-tool beneficial only for massive simulations (millions of edges).
"""

from collections import defaultdict
from heapq import *  # Binary heap for O(log V) priority queue operations
from copy import deepcopy

# Network topology: [from, to, base_distance]
# GRAPH REPRESENTATION ANALYSIS:
# - Uses adjacency list structure via defaultdict
# - Space efficient: O(V + E) vs O(V²) for adjacency matrix
# - Optimal for sparse graphs (typical in road networks)
# - Each edge stored as [source, dest, weight] tuple
distances = [['A','B',1], ['A','C',3], ['B','C',1], ['B','D',5], 
             ['C','B',2], ['C','E',1], ['D','E',7], ['D','F',2], 
             ['E','D',1], ['E','F',1]]

# Traffic distribution ratios at each node
# TRAFFIC MODEL EFFICIENCY:
# - Probabilistic distribution: O(1) lookup per node
# - Models realistic traffic flow patterns
# - Space: O(V * avg_destinations) - typically O(V)
distribution = {
    'A': [('A',0.5), ('B',0.35), ('C',0.15)],
    'B': [('B',0.4), ('C',0.2), ('D',0.4)],
    'C': [('C',0.9), ('B',0.1), ('E',0.1)],
    'D': [('D',0.3), ('E',0.4), ('F',0.3)],
    'E': [('E',0.7), ('D',0.1), ('F',0.2)],
    'F': [('F',1)]
}

# Initial car count at each node
# DYNAMIC STATE TRACKING:
# - O(1) access time for car counts
# - Updates in O(1) time per node
# - Alternative: Could use numpy arrays for vectorized operations
cars_initial = {'A':10, 'B':20, 'C':30, 'D':40, 'E':50, 'F':60}

# Congestion model: cars -> time penalty
# LOOKUP TABLE EFFICIENCY:
# - O(1) average access time
# - Small constant factor (9 entries)
# - Alternative: Mathematical formula could replace lookup
# - Current approach: Clear, maintainable, fast enough for use case
number_of_cars_time = {10:1, 20:2, 30:4, 40:8, 50:16, 60:32, 70:64, 80:128, 90:256}

def calc_weight(ele_dist, no_of_car):
    """
    Calculate edge weight based on distance and traffic congestion
    
    EFFICIENCY ANALYSIS:
    - Time Complexity: O(k) where k = number of congestion thresholds
    - Space Complexity: O(1) - constant space usage
    - Current k = 9 thresholds, so effectively O(1)
    
    CONGESTION MODEL:
    Exponential penalty system: 10 cars = 1x, 20 cars = 2x, ..., 60 cars = 32x
    Models realistic traffic congestion where density causes non-linear delays
    
    OPTIMIZATION OPPORTUNITIES:
    - Could use binary search for O(log k) lookup
    - Could precompute lookup table for O(1) access
    - Current linear search adequate for small threshold count
    
    Args:
        ele_dist: Edge data [source, destination, base_distance]
        no_of_car: Number of cars at the source node
    
    Returns:
        Weighted edge cost (base_distance + congestion_penalty)
    """
    t = 0
    # Linear search through congestion thresholds: O(k)
    for i in number_of_cars_time:
        if no_of_car < i:
            t = number_of_cars_time[i]
            break
    
    # Final weight = base distance + congestion penalty
    weight = ele_dist[2] + t
    return weight

def num_of_cars(node):
    """
    Calculate number of cars at a node after redistribution
    NOTE: This function has known bugs - see issue #1
    """
    t = 0
    for i in distances:
        if i[1] == node:
            for j in distribution:
                if j == i[0]:
                    for k in distribution[j]:
                        if k[0] == node:
                            t += (cars_initial[node]) * k[1]
    t -= (distribution[node][0][1]) * (cars_initial[node])
    cars_initial[node] = t
    return t

def dijkstra(edges, f, t):
    """
    Dijkstra's shortest path algorithm using binary heap priority queue
    
    EFFICIENCY ANALYSIS:
    - Time Complexity: O(E log V)
      * Each edge processed once: O(E)
      * heappush/heappop operations: O(log V) per operation
      * Total: O(E log V) where E = edges, V = vertices
    
    - Space Complexity: O(V + E)
      * Graph storage (adjacency list): O(V + E)
      * Priority queue: O(V) worst case
      * Distance tracking: O(V)
    
    IMPLEMENTATION DETAILS:
    - Uses heapq module for binary heap operations
    - Maintains seen set for visited vertices: O(1) lookup
    - Path reconstruction via tuple chaining
    
    LIBRARY ALTERNATIVES CONSIDERED:
    - NetworkX: dijkstra_path() - easier but slower due to Python overhead
    - graph-tool: shortest_path() - faster for massive graphs (>10^5 vertices)
    - igraph: shortest_paths() - C backend, good balance of speed/usability
    
    CURRENT CHOICE RATIONALE:
    For typical traffic networks (10-1000 vertices), stdlib implementation
    provides optimal balance of performance, transparency, and zero dependencies.
    
    Args:
        edges: List of [source, destination, weight] tuples
        f: Starting vertex
        t: Target vertex
    
    Returns:
        Tuple of (total_cost, path_as_nested_tuples) or float("inf") if no path
    
    NOTE: Bellman-Ford alternative needed for negative weights - see issue #5
    """
    # Build adjacency list representation: O(E) time, O(V + E) space
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    # Initialize priority queue with source vertex
    # heapq provides binary heap: O(log V) for push/pop operations
    q, seen, mins = [(0, f, ())], set(), {f: 0}
    
    while q:
        # Extract minimum cost vertex: O(log V)
        (cost, v1, path) = heappop(q)
        
        if v1 not in seen:
            seen.add(v1)  # Mark as visited: O(1)
            path = (v1, path)  # Reconstruct path via tuple chaining
            
            if v1 == t:  # Target reached
                return (cost, path)

            # Process all neighbors: O(degree(v1))
            for c, v2 in g.get(v1, ()):
                if v2 in seen:  # Skip visited vertices
                    continue
                    
                prev = mins.get(v2, None)
                next = cost + c
                
                # Relaxation step with heap insertion: O(log V)
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))
    
    return float("inf")  # No path found

def format_dijkstra(k):
    """Format dijkstra output to readable path string"""
    s = ''
    path_time = k[0]
    for i in str(k[1]):
        if i not in [',', '(', ')', "'", ' ']:
            s += i
    return (s[::-1], path_time)

def change_weights():
    """
    Update all edge weights based on current traffic
    
    EFFICIENCY ANALYSIS:
    - Time Complexity: O(E * V) where E = edges, V = vertices
    - For each edge (O(E)): looks up car count (O(1)) and calculates weight (O(1))
    - Space Complexity: O(1) - modifies existing data structure in-place
    
    OPTIMIZATION OPPORTUNITIES:
    - Current: Nested loops could be optimized
    - Alternative: Direct edge-to-node mapping for O(E) complexity
    - Current approach: Simple, readable, adequate for small graphs
    
    LIBRARY ALTERNATIVES:
    - NetworkX: set_edge_attributes() - more readable but slower
    - NumPy: Vectorized operations for large graphs
    - Current stdlib approach: Best balance for this use case
    """
    for i in distances:
        for j in cars_initial:
            if i[0] == j:  # Find matching source node
                i[2] = calc_weight(i, cars_initial[j])  # Update edge weight

# Main simulation
if __name__ == "__main__":
    start = input('Enter start point: ')
    end = input('Enter end point: ')
    
    # Calculate initial weights and best path
    change_weights()
    path = start
    print('The best route according to current traffic conditions.')
    k = format_dijkstra(dijkstra(distances, start, end))
    print('Route:', k[0], 'Time:', k[1])
    
    # Find initial edge weight for subject car
    for i in distances:
        if start == i[0] and k[0][1] == i[1]:
            t1 = i[2]
            break
    
    # Initialize event clock
    # NOTE: Data structure could be more efficient - see issue #3
    clock = []
    temp = deepcopy(distances)
    for i in temp:
        i.append(0)  # Add flag: 0 = other cars, 1 = subject car
    for i in temp:
        clock.append(i)
    clock.append([start, k[0][1], t1, 1])  # Subject car's first move
    
    # Simulation loop
    t = 1
    car_pos = start
    while car_pos != end:
        # Process events scheduled for time t
        for i in clock:
            # Regular cars arriving at nodes
            if i[2] == t and i[3] == 0:
                i[2] = (calc_weight(i[:3], num_of_cars(i[1]))) + t
                if i not in clock:
                    clock.append(i)
            
            # Subject car arriving at node
            if i[3] == 1 and i[2] == t:
                k = format_dijkstra(dijkstra(distances, car_pos, end))
                path += k[0][1]
                car_pos = k[0][1]
                if len(k[0]) == 2:
                    pass
                else:
                    clock.append([car_pos, k[0][2], k[1] + t, 1])
                print(path, t)
        
        # Remove processed events
        # NOTE: Removing items while iterating - potential bug! See issue #1
        for i in clock:
            if i[2] == t:
                clock.remove(i)
        t += 1
    
    print(path, t)
    print("\nSimulation complete!")
    print("NOTE: This is base code with known issues.")
    print("See GitHub issues for improvement tasks.")

"""
=== COMPREHENSIVE EFFICIENCY ANALYSIS ===

CURRENT IMPLEMENTATION SUMMARY:
┌──────────────────────┬─────────────────┬─────────────────────────────────┐
│ Component            │ Time Complexity │ Space Complexity                │
├──────────────────────┼─────────────────┼─────────────────────────────────┤
│ Dijkstra Algorithm   │ O(E log V)      │ O(V + E)                        │
│ Weight Calculation   │ O(1)            │ O(1)                            │
│ Traffic Redistribution│ O(E * V)        │ O(1)                            │
│ Graph Storage        │ O(E)            │ O(V + E)                        │
│ Overall Simulation   │ O(T * E log V)  │ O(V + E), T = time steps        │
└──────────────────────┴─────────────────┴─────────────────────────────────┘

STRENGTHS OF CURRENT APPROACH:
✓ Zero external dependencies - runs anywhere Python runs
✓ Transparent implementation - easy to understand and debug  
✓ Optimal complexity for small-medium graphs (V < 10³, E < 10⁴)
✓ Memory efficient - O(V + E) space for graph representation
✓ Fast binary heap operations via heapq - O(log V) per operation

WHEN TO CONSIDER ALTERNATIVES:

🔹 NetworkX (pip install networkx):
   - Use for: Prototyping, research, complex graph analysis
   - Benefits: Rich algorithm library, graph visualization, ease of use
   - Cost: 2-5x slower due to Python overhead
   - Migration: nx.dijkstra_path(G, source, target, weight='weight')

🔹 graph-tool (system dependencies required):
   - Use for: Massive graphs (V > 10⁵, E > 10⁶), performance-critical applications
   - Benefits: C++/Boost backend, 10-100x faster for large graphs
   - Cost: Complex installation, steeper learning curve
   - Migration: gt.shortest_path(g, source, target, weights=edge_weights)

🔹 igraph (pip install igraph):
   - Use for: High-performance needs, cross-language compatibility
   - Benefits: Fast C core, good balance of speed and usability
   - Cost: External dependency, less Python-native API
   - Migration: g.shortest_paths(source, target, weights='weight')

PERFORMANCE BENCHMARKS (estimated for this graph size):
- Current implementation: ~1ms per shortest path calculation
- NetworkX: ~2-5ms per calculation (Python overhead)
- graph-tool: ~0.1ms per calculation (C++ backend)
- igraph: ~0.2ms per calculation (C backend)

RECOMMENDATION:
For typical traffic simulations with < 1000 vertices, current stdlib 
implementation provides optimal balance of performance, maintainability, 
and zero external dependencies. Consider alternatives only when scaling 
to massive networks or when specialized graph algorithms are needed.

FUTURE OPTIMIZATIONS (maintaining stdlib approach):
1. Replace linear search in calc_weight() with binary search: O(log k)
2. Use more efficient event queue (binary heap): O(log n) vs O(n)
3. Cache shortest paths for repeated queries
4. Vectorize operations using numpy for large simulations
"""