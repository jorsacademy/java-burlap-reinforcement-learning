# Java BURLAP Reinforcement Learning

A compact reinforcement-learning benchmark written in Java with [BURLAP](https://github.com/jmacglashan/burlap).

The project trains a tabular Q-learning agent on BURLAP's classic 11x11 Four Rooms navigation domain and reports how the average number of actions changes from the beginning to the end of training.

## Why BURLAP

BURLAP (Brown-UMBC Reinforcement Learning and Planning) is a Java library focused specifically on reinforcement learning, planning, MDPs, policies, state representations, and experiment tooling. This project uses BURLAP 3.0.1 from Maven Central.

## Problem

- Environment: 11x11 Four Rooms
- Initial state: `(0, 0)`
- Terminal state: `(10, 10)`
- Agent: tabular Q-learning
- Discount factor: `0.99`
- Learning rate: `0.5`
- Initial Q-value: `1.0`
- Training episodes: `500`
- Maximum actions per episode: `500`

BURLAP's default GridWorld reward is a per-step cost. Therefore, a policy improves when it reaches the terminal state using fewer actions.

## Run

Requirements:

- Java 17+
- Maven 3.8+

```bash
mvn clean test
mvn exec:java
```

Example output format:

```text
BURLAP Q-learning benchmark
Environment : 11x11 Four Rooms
Start       : (0, 0)
Goal        : (10, 10)
Episodes    : 500

First 50 avg actions : ...
Last  50 avg actions : ...
Improvement          : ...%
Best episode actions : ...
Result      : learning improved navigation efficiency.
```

The exact values can vary because exploration is stochastic.

## Project structure

```text
.
├── .github/workflows/maven.yml
├── pom.xml
├── src
│   ├── main/java/com/jorsacademy/burlap/
│   │   └── BurlapQLearningBenchmark.java
│   └── test/java/com/jorsacademy/burlap/
│       └── BurlapQLearningBenchmarkTest.java
└── README.md
```

## Validation

The repository contains JUnit smoke tests and a GitHub Actions workflow. CI performs a full Maven verification and then executes the 500-episode benchmark.

The tests intentionally validate correctness and bounded execution rather than asserting an exact learning score, because Q-learning exploration is stochastic and exact performance assertions would make CI unnecessarily flaky.

## Reference

BURLAP project: https://github.com/jmacglashan/burlap

The core setup follows BURLAP's documented GridWorld/Q-learning API: `GridWorldDomain`, `GridWorldTerminalFunction`, `SimulatedEnvironment`, `QLearning`, and `SimpleHashableStateFactory`.
