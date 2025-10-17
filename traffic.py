"""
Traffic Simulation - Base Version for Issue Assignments
This code works but has known issues for participants to fix/improve
"""

from collections import defaultdict
from heapq import *  # Uses heapq (binary heap) for priority queue operations
from copy import deepcopy


# Network topology: [from, to, base_distance]

effi_analysis = """
Efficiency Analysis of Current Implementation:

Algorithm: Dijkstra shortest path using heapq (we use the binary heap tho)
data Structures Used: heapq for the priority queue, defaultdict(list) for the adjacency list.

Time Complexity: O(E log V)
since each edge is considered once, and each relaxation involves a heap operation, the total cost becomes O(log V) each.

Space Complexity: O(V + E)
graph storage (adjacency list) plus the distance and arrays that came before.

Summary: standard efficiency for Dijkstra with a binary heap. suitable for small to medium graphs, 
but slower than compiled libraries for large datasets.good for these smaller use cases but not recommended for larger cases
"""

distances = [['A','B',1], ['A','C',3], ['B','C',1], ['B','D',5], 
             ['C','B',2], ['C','E',1], ['D','E',7], ['D','F',2], 
             ['E','D',1], ['E','F',1]]

# Traffic distribution ratios at each node
distribution = {
    'A': [('A',0.5), ('B',0.35), ('C',0.15)],
    'B': [('B',0.4), ('C',0.2), ('D',0.4)],
    'C': [('C',0.9), ('B',0.1), ('E',0.1)],
    'D': [('D',0.3), ('E',0.4), ('F',0.3)],
    'E': [('E',0.7), ('D',0.1), ('F',0.2)],
    'F': [('F',1)]
}
docu_analysis = """
Implementation Documentation and Comments:

uses heapq for priority queue operations.
each push/pop is O(log V), leading to total time complexity O(E log V).
space complexity O(V + E) for adjacency list and tracking the distance .


"""


# Initial car count at each node
cars_initial = {'A':10, 'B':20, 'C':30, 'D':40, 'E':50, 'F':60}

# Congestion model: cars -> time penalty
number_of_cars_time = {10:1, 20:2, 30:4, 40:8, 50:16, 60:32, 70:64, 80:128, 90:256}

def calc_weight(ele_dist, no_of_car):
    """Calculate edge weight based on distance and traffic congestion"""
    t = 0
    for i in number_of_cars_time:
        if no_of_car < i:
            t = number_of_cars_time[i]
            break
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
    Dijkstra's shortest path algorithm
    NOTE: Bellman-Ford alternative needed - see issue #5
    """
    g = defaultdict(list)
    for l, r, c in edges:
        g[l].append((c, r))

    q, seen, mins = [(0, f, ())], set(), {f: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t:
                return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                prev = mins.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))
    return float("inf")

def format_dijkstra(k):
    """Format dijkstra output to readable path string"""
    s = ''
    path_time = k[0]
    for i in str(k[1]):
        if i not in [',', '(', ')', "'", ' ']:
            s += i
    return (s[::-1], path_time)

def change_weights():
    """Update all edge weights based on current traffic"""
    for i in distances:
        for j in cars_initial:
            if i[0] == j:
                i[2] = calc_weight(i, cars_initial[j])
                

def bellman_ford(edges,start,end):
    # collect nodes
    nodes = set()
    for u, v, w in edges:
        nodes.add(u)
        nodes.add(v)

    if start not in nodes or end not in nodes:
        return float("inf")

    # initialize distances and predecessors
    dist = {n: float('inf') for n in nodes}
    prev = {n: None for n in nodes}
    dist[start] = 0

 
    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break

    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return float("inf")

    if dist[end] == float('inf'):
        return float("inf")
    
    rev = []
    node = end
    while node is not None:
        rev.append(node)
        node = prev.get(node)
    if rev[-1] != start:
        return float("inf")
    path_nodes = rev[::-1]  

    nested = ()
    for n in path_nodes:
        nested = (n, nested)

    return (dist[end], nested)
library_comparison = """
Comparison with Graph Libraries:

1. Stdlib (heapq + defaultdict)
    Benefits: No external dependencies, transparent, good up to lower no. of edges.
    Trade-offs: Slower for massive graphs, manual path reconstruction.

2. NetworkX
    Benefits: Easy to use, rich algorithm support, great for learning and prototyping.
    Trade-offs: Very slow for large graphs due to pure Python implementation.

3. igraph
    Benefits: C core, fast, scalable, multi-language (Python,C)
    Trade-offs: External dependency, less Pythonic API(who cares tho).

4. graph-tool
    Benefits: Extremely fast (C++/Boost backend), handles huge graphs.
    Trade-offs: Difficult installation, complex API, suited for research-level performance.
"""


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
benchmark_summary = """
benchmark summary (approx):

Library              1K/10K Edges  10K/100K Edges 100K/1M Edges(edges or no of connections btw vertices) 

Stdlib (heapq)        ~0.02 s       ~0.5 s          ~6 s          
NetworkX              ~0.1 s        ~3 s            ~>60 s        
igraph                ~0.01 s       ~0.2 s          ~1 s          
graph-tool            ~0.005 s      ~0.1 s          ~0.6 s        

(values are taken from https://www.ultipa.com/article/benchmarks and stackoverflow )



final analysis:for large or performance-critical workloads, consider compiled libraries such as igraph,
which can offer 5 to 10x speed improvements due to optimized Cpp backends.

"""
