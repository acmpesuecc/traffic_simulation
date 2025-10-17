# Traffic Simulation

Dynamic traffic routing system that recalculates optimal paths based on real-time traffic conditions.

## What it does

- Simulates traffic flow in a road network using directed graphs
- Recalculates shortest paths dynamically as traffic conditions change
- Models congestion with exponential time penalties based on car density
- Demonstrates limitations of static navigation systems

## Tech Stack

- Python 3
- Graph theory and shortest path algorithms (Dijkstra)
- Standard library: collections, heapq, copy

## Setup and Installation

Clone the repository:
```bash
git clone https://github.com/acmpesuecc/traffic_simulation
cd traffic_simulation
```

Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the simulation:
```bash
python traffic.py
```

## Usage

When prompted:
- Enter start point (e.g., A)
- Enter end point (e.g., F)
- Observe dynamic route recalculation as traffic redistributes

## Project Structure

```
traffic_simulation/
├── traffic.py          # Main simulation script
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── CONTRIBUTING.md    # Contribution guidelines
└── LICENSE            # Open source license
```

## Known Issues

Current implementation has several improvement opportunities:
- Bug in traffic propagation algorithm
- Missing Bellman-Ford alternative
- No graph visualization
- Inefficient data structures
- Limited library usage

See open issues for contribution opportunities.

## Maintainers

- [Rex-8](https://github.com/Rex-8)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up development environment
- Submitting bug reports
- Proposing enhancements
- Creating pull requests

## License

This project is open source. See LICENSE file for details.