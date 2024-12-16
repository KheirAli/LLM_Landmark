# LLM-Augmented Symbolic RL with Landmark-Based Task Decomposition

This repository provides code and instructions for running LLM-augmented symbolic Reinforcement Learning (RL) experiments with landmark-based task decomposition. The overall workflow involves:

1. Running an environment to collect trajectories.
2. Applying a landmark detection algorithm to the collected trajectories.
3. Using an LLM to generate refined rules based on the detected landmarks.

## Repository Structure

- **games/**  
  Code and scripts for running the environment and gathering trajectories. This is the first step in the workflow, where you interact with the environment to collect both **positive** and **negative** trajectories.

- **landmark-detection/**  
  Code and algorithms for landmark detection. Once you have your collected trajectories, you apply the landmark detection algorithm to identify key events or "landmarks" that help decompose the task into smaller subproblems.

- **LLM-rules/**  
  Code and instructions for prompting a Large Language Model (LLM) to generate refined symbolic rules. These rules leverage the landmarks and the structure of your environment to guide the RL agent with symbolic constraints or hints.

