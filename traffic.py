"""
Traffic Simulation - Final and Fully Corrected Version (Flow Conservation Model)

This code incorporates the robust flow conservation model (New = Current - Outgoing + Incoming)
to correctly manage car counts across simulation steps, and fixes the common 
list-modification runtime error.
"""

from collections import defaultdict
from heapq import *
from copy import deepcopy
import sys
# Set a recursion limit for safety, especially with pathfinding algorithms
sys.setrecursionlimit(2000) 


# --- DATA STRUCTURES ---


# Network topology: [from, to, base_distance]
distances = [['A','B',1], ['A','C',3], ['B','C',1], ['B','D',5], 
             ['C','B',2], ['C','E',1], ['D','E',7], ['D','F',2], 
             ['E','D',1], ['E','F',1]]


# Traffic distribution ratios at each node
distribution = {
    'A': [('A',0.5), ('B',0.35), ('C',0.15)],
    'B': [('B',0.4), ('C',0.2), ('D',0.4)],
    'C': [('C',0.8), ('B',0.1), ('E',0.1)],
    'D': [('D',0.3), ('E',0.4), ('F',0.3)],
    'E': [('E',0.7), ('D',0.1), ('F',0.2)],
    'F': [('F',1.0)]
}


# Initial car count at each node
cars_initial = {'A':10, 'B':20, 'C':30, 'D':40, 'E':50, 'F':60}


# Congestion model: cars -> time penalty
number_of_cars_time = {10:1, 20:2, 30:4, 40:8, 50:16, 60:32, 70:64, 80:128, 90:256}


# --- TRAFFIC & WEIGHT FUNCTIONS ---


def calc_weight(ele_dist, no_of_car):
    """Calculate edge weight based on distance and traffic congestion"""
    t = 0
    for threshold, penalty in sorted(number_of_cars_time.items()):
        if no_of_car < threshold:
            t = penalty
            break
    if t == 0:
        t = list(number_of_cars_time.values())[-1]
        
    weight = ele_dist[2] + t
    return weight


def calculate_traffic_flow():
    """
    Calculates the NEW car count for ALL nodes based on the Flow Conservation Model:
    New = Current - Outgoing + Incoming. Returns a new state dictionary.
    """
    global cars_initial, distribution, distances
    
    all_nodes = set(cars_initial.keys())
    for frm, to, _ in distances:
        all_nodes.add(frm)
        all_nodes.add(to)
        
    new_cars = {node: 0.0 for node in all_nodes}
    
    # 1. Calculate Outgoing flow for all nodes
    outgoing_flow = defaultdict(float)
    for source_node, ratios in distribution.items():
        current_count = cars_initial.get(source_node, 0)
        
        total_outgoing = 0.0
        for dest, ratio in ratios:
            if dest != source_node:
                total_outgoing += current_count * ratio
        
        outgoing_flow[source_node] = total_outgoing
        
    # 2. Calculate Incoming flow and the New count for all nodes
    for node in all_nodes:
        current_count = cars_initial.get(node, 0)
        
        # Calculate total incoming flow from *other* nodes
        incoming_flow = 0.0
        for source_node, ratios in distribution.items():
            if source_node != node:
                if any(e[0] == source_node and e[1] == node for e in distances):
                    for dest, ratio in ratios:
                        if dest == node:
                            incoming_flow += cars_initial.get(source_node, 0) * ratio
                            break
        
        # Stay-Put Count = Current - Total Outgoing
        stay_put_count = current_count - outgoing_flow[node]
        
        # Final New Count = Stay-Put + Incoming
        new_count = max(0.0, stay_put_count + incoming_flow)
        
        new_cars[node] = new_count

    return new_cars


def dijkstra(edges, f, t):
    """Dijkstra's shortest path algorithm"""
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
                next_cost = cost + c
                if prev is None or next_cost < prev:
                    mins[v2] = next_cost
                    heappush(q, (next_cost, v2, path))
    return float("inf"), None


def format_dijkstra(k):
    """Format dijkstra output to readable path string"""
    if k[1] is None:
        return ("No Path", k[0])
        
    def path_to_list(nested_tuple):
        path = []
        while nested_tuple:
            node, nested_tuple = nested_tuple
            path.append(node)
        return path[::-1]

    path_time = k[0]
    path_list = path_to_list(k[1])
    path_str = "".join(path_list)
    
    return (path_str, path_time)


def change_weights():
    """Update all edge weights based on current traffic at the source node"""
    for i in distances:
        source_node = i[0]
        current_cars = cars_initial.get(source_node, 0)
        i[2] = calc_weight(i, current_cars)
                

# --- MAIN SIMULATION LOOP (Refactored for Clean State Updates) ---


if __name__ == "__main__":
    start = input('Enter start point: ').strip().upper()
    end = input('Enter end point: ').strip().upper()
    
    if start not in cars_initial or end not in cars_initial:
        print("Error: Start or end point not in network.")
        exit()

    # Calculate initial weights and best path
    change_weights()
    k = format_dijkstra(dijkstra(distances, start, end))
    
    if k[0] == "No Path":
        print(f"Error: No path found from {start} to {end}.")
        exit()
        
    path_str = k[0]
    print('The best route according to current traffic conditions.')
    print(f'Route: {path_str}, Initial Time: {k[1]}')
    
    path_taken = start
    
    # Find initial edge weight for subject car
    t1 = -1
    next_node_initial = path_str[1]
    for i in distances:
        if start == i[0] and next_node_initial == i[1]:
            t1 = i[2]
            break
    
    if t1 == -1:
        print(f"Error: Cannot find initial segment weight from {start} to {next_node_initial}. Exiting.")
        exit()

    # Initialize event clock
    clock = []
    
    # 1. Schedule "Other Car" movements (simplified flow)
    for i in distances:
        source, dest, weight = i
        clock.append([source, dest, weight, 0]) 
    
    # 2. Schedule Subject Car's first move
    clock.append([start, next_node_initial, t1, 1]) 
    
    # Simulation loop
    t = 1
    car_pos = start
    MAX_TIME = 2000 # Safety break
    
    while car_pos != end and t < MAX_TIME:
        events_to_remove = []
        clock.sort(key=lambda x: x[2]) 
        
        # --- 1. Process Events Scheduled for Time t ---
        for i in clock:
            if i[2] == t:
                source, dest, arrival_time, is_subject_car = i
                
                if is_subject_car == 0:
                    # Regular cars arrive. Schedule the next segment flow.
                    
                    # Schedule the next representative flow movement out of 'dest'
                    next_dest = None
                    max_ratio = 0
                    for d, ratio in distribution.get(dest, []):
                        if d != dest and ratio > max_ratio:
                            max_ratio = ratio
                            next_dest = d
                            
                    if next_dest and max_ratio > 0:
                        next_edge = [e for e in distances if e[0] == dest and e[1] == next_dest]
                        if next_edge:
                            # Use current traffic at 'dest' for weight calculation
                            next_weight = calc_weight(next_edge[0], cars_initial.get(dest, 0))
                            clock.append([dest, next_dest, next_weight + t, 0])
                            
                else:  # is_subject_car == 1
                    # Subject car arrives at 'dest'
                    car_pos = dest
                    
                    if car_pos == end:
                        path_taken += car_pos
                        print(f"Final Arrival: Route {path_taken}, Total Time: {t}")
                        events_to_remove.append(i)
                        break
                    
                    # Recalculate best route based on current traffic
                    change_weights()  # Update all edge weights based on global state
                    k = format_dijkstra(dijkstra(distances, car_pos, end))
                    
                    if k[0] == "No Path":
                        print(f"Error: No path found from current position {car_pos} to {end}. Simulation halted.")
                        car_pos = end
                        break

                    next_dest = k[0][1]

                    # Find the segment weight for the move
                    next_weight = -1
                    for edge in distances:
                        if edge[0] == car_pos and edge[1] == next_dest:
                            next_weight = edge[2]
                            break
                    
                    if next_weight > 0:
                        new_arrival_time = t + next_weight
                        clock.append([car_pos, next_dest, new_arrival_time, 1])
                        path_taken += next_dest
                        print(f"Move at t={t}: From {source} to {dest}. New position: {car_pos}. Next segment weight: {next_weight}. Path: {path_taken}")
                    else:
                        print(f"Error: Could not find next segment weight from {car_pos} to {next_dest}. Simulation halted.")
                        car_pos = end
                        break
                
                events_to_remove.append(i)
            
        # --- 2. Update Global State (Traffic Propagation) ---
        # Commit the traffic changes after all time=t events have been processed.
        new_cars_state = calculate_traffic_flow()
        cars_initial = new_cars_state
        
        # Safely remove processed events after the iteration
        clock = [event for event in clock if event not in events_to_remove]
                
        if car_pos == end:
            break
            
        t += 1
        
    print("-" * 40)
    if t >= MAX_TIME:
        print(f"Simulation ended at max time limit (t={MAX_TIME}).")
    elif car_pos == end:
        print("Simulation complete! Subject car reached the destination.")
    else:
        print("Simulation stopped prematurely.")
    print(f"Final Route: {path_taken}")
    print(f"Final Total Time: {t}")
