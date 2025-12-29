
# Self-Improving LLM Agent for Puzzle Solving

## Project Overview

This project presents the design and implementation of a self-improving Large Language Model (LLM) agent capable of solving logic-based puzzles through iterative reasoning, self-reflection, and adaptive behavior. The agent is evaluated on structured problem domains such as logical riddles, constraint-based puzzles (e.g., Sudoku), and text-based reasoning challenges. Unlike static prompt-based systems, this agent continuously improves its performance by leveraging feedback signals derived from success and failure over large-scale experimental runs.

The primary objective of this project is to explore how LLMs can be engineered as autonomous agents rather than passive text generators, focusing on reasoning quality, error analysis, and long-term performance improvement.

## Motivation

While modern LLMs demonstrate impressive zero-shot and few-shot reasoning capabilities, they often fail in tasks requiring multi-step logical consistency, long-horizon planning, or error recovery. This project addresses these limitations by introducing a closed-loop agent architecture that incorporates self-reflection, memory, and feedback-driven adaptation. The work is motivated by real-world LLM engineering challenges, where robustness, evaluability, and continuous improvement are essential.

## System Architecture

The system is composed of a modular agent framework with clearly separated responsibilities. At its core, a baseline LLM agent attempts to solve puzzles using structured prompts. Surrounding this core are additional modules responsible for evaluation, reflection, memory storage, and strategy adaptation. A multi-agent debate layer extends the architecture by allowing multiple specialized sub-agents to propose competing solutions, which are then analyzed and aggregated by a decision module.

This modular design ensures extensibility, enabling independent experimentation with reflection strategies, reward definitions, and agent roles without altering the entire system.

## Self-Reflection and Learning Mechanism

After each puzzle attempt, the agent performs a self-reflection step in which it analyzes its own reasoning process and outcome. Errors, incorrect assumptions, and failed reasoning paths are summarized in a structured format and stored in an external memory. This memory is subsequently used to influence future attempts by adjusting prompts, prioritizing successful reasoning strategies, and avoiding previously identified failure patterns.

Learning is driven by explicit feedback signals derived from task outcomes. These signals function as lightweight reinforcement cues, guiding the agent’s adaptive behavior over time without requiring full end-to-end model retraining.

## Multi-Agent Debate Mode

The system includes an optional multi-agent debate mechanism in which multiple sub-agents with distinct reasoning styles independently attempt to solve the same puzzle. For example, different agents may emphasize conservative logical deduction, exploratory reasoning, or critical evaluation. A centralized arbitration process compares these candidate solutions and selects the final answer based on predefined criteria.

This approach allows the system to exploit diversity in reasoning paths and has been shown to reduce logical inconsistencies and improve overall solution quality on complex puzzles.

## Evaluation and Metrics

Performance is evaluated using quantitative and qualitative metrics, including success rate, number of reasoning steps, frequency of self-corrections, and improvement trends across large puzzle sets. A baseline agent without self-improvement or debate mechanisms is used for comparison, allowing the impact of each architectural component to be measured independently.

Experiments are conducted over thousands of puzzle instances to assess scalability, stability of learning, and long-term behavioral trends.

## Project Scope and Relevance

This project is positioned as an applied LLM Engineering effort rather than a purely theoretical exploration. It demonstrates practical system design choices, evaluation methodologies, and architectural patterns relevant to real-world agent-based LLM applications, including automated reasoning systems, decision-support tools, and autonomous AI agents.

## Limitations and Future Work

Current limitations include reliance on prompt-level adaptation rather than parameter-level learning, and the use of task-specific reward definitions. Future extensions may include curriculum learning, more formal reinforcement learning integration, dynamic agent role evolution, and transfer of learned strategies across different puzzle domains.

## Project structure

```text
self-improving-llm-agent/
│
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── configs/
│   ├── agent.yaml
│   ├── prompts.yaml
│   └── evaluation.yaml
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── reflective_agent.py
│   │   ├── debate_agent.py
│   │   └── arbitration.py
│   │
│   ├── reasoning/
│   │   ├── solver.py
│   │   ├── reflection.py
│   │   └── strategies.py
│   │
│   ├── memory/
│   │   ├── episodic_memory.py
│   │   ├── vector_memory.py
│   │   └── memory_store.py
│   │
│   ├── feedback/
│   │   ├── reward.py
│   │   └── adaptation.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── baseline.py
│   │   └── experiment_runner.py
│   │
│   ├── puzzles/
│   │   ├── puzzle.py
│   │   ├── sudoku.py
│   │   └── logic_riddles.py
│   │
│   ├── llm/
│   │   ├── client.py
│   │   └── prompt_builder.py
│   │
│   └── utils/
│       ├── logging.py
│       └── helpers.py
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_reflective.py
│   └── run_debate.py
│
├── data/
│   ├── puzzles/
│   ├── logs/
│   └── results/
│
├── notebooks/
│   └── analysis.ipynb
│
└── tests/
    ├── test_agents.py
    ├── test_memory.py
    └── test_evaluation.py

```
