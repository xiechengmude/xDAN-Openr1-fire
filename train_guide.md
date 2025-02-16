# GRPO Training Guide

## Overview
This guide explains the GRPO (Generative Reinforcement Policy Optimization) training implementation for the xDAN-Openr1-fire project. The training process is designed to enhance the model's reasoning capabilities using multiple reward functions and a structured conversation format.

## Key Components

### 1. Training Configuration
The training process uses three main configuration classes:
- `GRPOScriptArguments`: Custom training arguments
- `GRPOConfig`: GRPO-specific configuration
- `ModelConfig`: Base model configuration

### 2. Reward Functions
Multiple reward functions are available and can be combined:

| Reward Function | Description |
|----------------|-------------|
| `accuracy_reward` | Evaluates answer correctness |
| `format_reward` | Checks response format compliance |
| `reasoning_steps_reward` | Assesses quality of reasoning steps |
| `cosine_scaled_reward` | Provides scaled rewards based on cosine similarity |
| `repetition_penalty_reward` | Penalizes repetitive content |
| `len_reward` | Considers response length |

### 3. Conversation Format
The training uses a structured conversation format:
```
System Prompt: Defines the conversation structure between User and Assistant
Format: 
<think> reasoning process </think>
<answer> final answer </answer>
```

## Training Process

1. **Initialization**
   - Sets random seed for reproducibility
   - Configures logging
   - Checks for existing checkpoints
   - Initializes WandB if enabled

2. **Dataset Preparation**
   - Loads dataset using HuggingFace's datasets library
   - Formats data into conversation structure
   - Removes unnecessary columns

3. **Model Setup**
   - Configures model parameters (dtype, attention implementation)
   - Sets up gradient checkpointing if enabled
   - Initializes GRPO trainer

## Configuration Options

### Reward Function Parameters
- Cosine Scaling:
  - `cosine_min_value_wrong`: Minimum reward for incorrect answers (default: 0.0)
  - `cosine_max_value_wrong`: Maximum reward for incorrect answers (default: -0.5)
  - `cosine_min_value_correct`: Minimum reward for correct answers (default: 0.5)
  - `cosine_max_value_correct`: Maximum reward for correct answers (default: 1.0)
  - `cosine_max_len`: Maximum length for scaling (default: 1000)

- Repetition Penalty:
  - `repetition_n_grams`: N-gram size (default: 3)
  - `repetition_max_penalty`: Maximum penalty (default: -1.0)

## Best Practices

1. **Reward Function Selection**
   - Start with basic rewards (`accuracy` and `format`)
   - Add additional rewards based on specific requirements
   - Monitor training metrics to adjust reward combinations

2. **Training Stability**
   - Use gradient checkpointing for large models
   - Enable WandB logging for monitoring
   - Regularly save checkpoints

3. **Performance Optimization**
   - Adjust batch size based on available GPU memory
   - Use mixed precision training when possible
   - Enable gradient accumulation for larger effective batch sizes

## Monitoring and Debugging

1. **Logging**
   - Training progress is logged to stdout
   - WandB integration available for metric tracking
   - Checkpoint management for training resumption

2. **Key Metrics**
   - Monitor reward components individually
   - Track overall model performance
   - Observe convergence patterns

## References
- GRPO implementation based on HuggingFace's TRL library
- Follows LIMO (Less is More for Reasoning) principles
- Utilizes standard HuggingFace training utilities
