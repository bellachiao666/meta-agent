# Meta-Agent Project Scaffold

This repository hosts a reinforcement-learning-driven orchestration layer for a team of specialized coding agents. The current scaffold provides placeholders for configuration, agent implementations, environment wrappers, RL components, and supporting utilities. Populate each module as your project evolves.

## Directory Overview
- `config/`: YAML configuration for model routing and RL hyperparameters.
- `data/`: Sample tasks, related test fixtures, and execution logs.
- `agents/`: Individual agent implementations (coder, tester, reviewer, meta-controller).
- `env/`: Environment abstractions exposing state, action, and reward hooks for the meta-agent.
- `rl/`: Policy definitions and training loop utilities for RL-based coordination.
- `scripts/`: Executable entry points for running pipelines, training, and evaluation.
- `utils/`: Shared logging helpers and prompt templates.
- `tests/`: End-to-end validation suite.

Each source file currently includes a brief description to guide future implementation.
