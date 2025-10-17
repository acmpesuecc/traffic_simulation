# Contributing to Traffic Simulation

Guidelines for contributing to the Traffic Simulation project hosted by ACM PES University, EC Campus.

## Prerequisites

- Python 3.x installed
- Understanding of graph theory and shortest path algorithms
- Familiarity with Git and GitHub workflow

## Setup for Development

1. Fork and clone the repository:
```bash
git clone https://github.com/acmpesuecc/traffic_simulation
cd traffic_simulation
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a new branch for your work:
```bash
git checkout -b feature/your-feature-name
```

## How to Contribute

### Bug Reports

When filing a bug report, include:
- Clear, descriptive title identifying the problem
- Exact steps to reproduce the issue
- Expected vs actual behavior
- Screenshots (if applicable)
- Your environment (OS, Python version)

Example:
```
Title: Traffic redistribution fails at node C
Steps: Run simulation from A to F, observe output at t=5
Expected: Cars redistribute according to distribution ratios
Actual: Car count becomes negative
```

### Feature Enhancements

When proposing enhancements, include:
- Clear, descriptive title
- Detailed description of the proposed feature
- Step-by-step implementation approach
- Use cases and benefits
- Similar implementations in other projects (if any)

### Pull Requests

Before submitting a PR:
- Ensure code runs without errors
- Test with multiple start/end node combinations
- Verify changes don't break existing functionality
- Follow the existing code style

PR checklist:
- [ ] Code tested locally
- [ ] Descriptive commit messages (present tense)
- [ ] No unnecessary files included
- [ ] Updated documentation (if needed)
- [ ] References related issue number

## Coding Conventions

- Use descriptive variable names
- Add comments for complex logic
- Follow PEP 8 style guidelines
- Keep functions focused and modular
- Commit messages in present tense (e.g., "Add Bellman-Ford implementation")

## Issue Labels

- `bug` - Something isn't working correctly
- `enhancement` - New feature or improvement
- `documentation` - Documentation improvements
- `good first issue` - Beginner-friendly tasks
- `bounty~X` - Issue difficulty/points indicator

## Questions

For queries or clarifications, contact:
- [Rex-8](https://github.com/Rex-8)

## Code Review Process

Maintainers will:
- Review PRs within reasonable timeframe
- Provide constructive feedback
- Request changes if needed
- Merge once all requirements are met

Thank you for contributing to Traffic Simulation!