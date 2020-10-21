# traffic_simulation
## Domain name: Traffic management in urban areas
## Specifics: 
Overcrowding of roads due to people referring to navigational sites and a majority of them using the same routes.
## Abstract: 
The traffic on our roads has increased manifold over the last couple of decades
especially in urban areas. As a result, much time and energy are wasted on travelling. With the
advancement in technology, various navigation applications have gained come up, the most
popular of these being-Google Maps.
Google Maps provides information about various routes and real-time traffic. However, what
actually happens is that a large number of people look for the same route on Google Maps at the
same point of time and they all choose whichever is given as the fastest route. Because of this,
there is an exponential increase in traffic on the supposedly fastest route.
Our idea is an improvement on Google Maps which counts the number of users using the same
route at the same time, then estimates the increase in traffic if a majority of them takes this same
route. If there is a substantial increase in traffic, it informs the user about the same and advices that
he could take the second-best route instead. This tweak in the system would remove one of the
most basic flaws in today’s navigation applications.
Hence, lesser time is spent in traffic and the destinations feel closer.
## Keywords: traffic, application, route, Google Maps
## Technologies Used:
- Python
- Graph theory
- Dijkstra's algorithm

## Assumptions Made:
- Directed graphs are used represent road networks
- Best path computation in road networks is modelled as a shortest path problem in directed graphs
- Uses random graph generation as inputs to the prototype
-> Our implementation of random graph generation results only in directed acyclic graphs
- Extending this to generate cyclic graphs is straight forward
- Our model of car movement is discrete
-> Can lead to certain unrealistic situations

## Algorithm
1. A random graph is generated to simulate road networks
2. Random values are generated for the time required to travel between two nodes, the current traffic on these roads and the fractions in which traffic will divide at the junctions.
3. The time taken to travel through each edge is calculated in current traffic conditions.  The quickest route to go to the destination is determined.
4. The traffic conditions after the stipulated time are recalculated.
5. Determine the quickest path  and the time taken in the new conditions.
6. This path will be the true ideal route to the destination.

## Propogation of Updates
1. apply initial weights according to current number of cars 
2. current_node_of_subject_car = A ; destination_node= J
3. initial_path = single_source_shortest_path_Dijkstra(source_node: A)
4. T = 0; q = φ; final_path = {A}; time_taken_in_final_path = 0
5. for every edge e=(u,v) ϵ set_of_edges_in_graph
          add <v, W> to q where W is the weight on edge e
6. while q != φ

           6.1. remove all entries <u, W> from q such that W == T
           6.2. for all edges (u, x) in the set_of_edges_in_graph
                   6.2.1. move cars from u to x according the traffic distribution ratios obtained from distribution table
                   6.2.2. update weights on all edges (p, u) and (x,q)
           6.3. new_path = single_source_shortest_path_Dijkstra(current_node_of_subject_car)
           6.4. let first edge in new_path be (current_node_of_subject_car, y) with weight r
           6.5. current_node_of_subject_car = y
           6.6. final_path = final_path ∪ (current_node_of_subject_car, y)
           6.7. time_taken_in_final_path = time_taken_in_final_path + r
           6.8. If (y != destination_node)
                      6.8.1. T++ 
                      6.8.2. for every edge e=(u,v) ϵ set_of_edges_in_graph
                                6.8.2.1. add <v, T+W> to q where W is the weight on edge e
7. output final_path and time_taken_in_final_path

## Improvements
- Considering other parameters than the quickest path
- Real-world traffic model
- To reduce run-time we can try to incorporate an incremental version of Dijkstra’s algorithm
- Ruling out obvious longer paths by other means
- More user-friendly interface
