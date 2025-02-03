# Group Relative Policy Optimization (GRPO)

GRPO is a key innovation in DeepSeek-R1 that simplifies and improves upon traditional PPO for large language model training. This document explains its core concepts and implementation details.

## Overview

GRPO is designed to be more efficient and effective than PPO for training large language models, particularly for tasks requiring complex reasoning. Here are its key features:

1. **No Value Function Model**: Unlike PPO, GRPO eliminates the need for a separate value function model, reducing memory usage and computational overhead.

2. **Group-based Advantage Computation**: For each input, GRPO uses a group of outputs and computes the baseline reward as the average score within that group. This approach aligns well with reward model training.

3. **Direct KL Divergence Optimization**: Instead of incorporating KL divergence into the reward signal like PPO, GRPO directly integrates it into the loss function for finer control during optimization.

## Implementation Details

The core GRPO implementation in `src/open_r1/grpo.py` consists of:

### 1. Reward Functions

```python
def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion matches the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        try:
            answer = parse(content)
            reward = float(verify(answer, parse(sol)))
        except Exception:
            reward = 0.0
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion follows the required format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

### 2. Training Process

The training process involves:
1. Generating multiple outputs for each input
2. Computing rewards using registered reward functions
3. Normalizing rewards within groups
4. Optimizing the policy with KL divergence regularization

## Mathematical Formulation

GRPO's objective function can be expressed as:

```
L_GRPO(θ) = E[r_g(a) * log(π_θ(a|s))] - β * KL(π_θ || π_ref)
```

where:
- r_g(a) is the group-normalized reward
- π_θ is the current policy
- π_ref is the reference policy
- β is the KL penalty coefficient

## Usage

To train a model using GRPO:

```bash
python -m src.open_r1.grpo \
    --model_name_or_path deepseek-ai/deepseek-base \
    --dataset_name math_dataset \
    --reward_funcs accuracy format \
    --output_dir output/grpo_math
```

## Best Practices

1. **Reward Function Design**:
   - Keep reward functions simple and targeted
   - Combine multiple reward signals when needed
   - Normalize rewards within groups

2. **Training Stability**:
   - Start with a well-supervised model
   - Monitor KL divergence carefully
   - Use appropriate learning rates and batch sizes

3. **Hyperparameter Tuning**:
   - Adjust the KL penalty coefficient (β)
   - Tune the group size for advantage computation
   - Balance different reward components

---

# GRPO 组相对策略优化

## 核心概念

GRPO（组相对策略优化）是DeepSeek R1的核心创新，通过以下方式提升模型训练效率：

### 关键特性
1. **无价值模型**：相比PPO，省去了独立的价值模型
2. **组优势计算**：基于组内输出的平均奖励进行归一化
3. **直接KL优化**：在损失函数中直接集成KL散度约束

## 实现细节

### 1. 奖励函数实现
```python
def accuracy_reward(completions, solution):
    # 检查答案与标准解的匹配度
    ...

def format_reward(completions):
    # 验证输出格式是否符合规范
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    ...
```

### 2. 训练流程
1. 为每个输入生成多个输出
2. 使用注册的奖励函数计算奖励
3. 组内奖励归一化
4. 带KL散度正则化的策略优化

## 数学公式

GRPO目标函数：
```
L_GRPO(θ) = E[r_g(a) * log(π_θ(a|s))] - β * KL(π_θ || π_ref)
```
其中：
- r_g(a): 组归一化奖励
- π_θ: 当前策略
- π_ref: 参考策略
- β: KL惩罚系数

## 最佳实践
1. **奖励设计**：
   - 保持奖励函数简单明确
   - 必要时组合多个奖励信号
   - 进行组内归一化

2. **训练稳定性**：
   - 从良好监督的模型开始
   - 密切监控KL散度
   - 使用合适的学习率和批量大小

## 完整中文版见 [grpo_zh.md](grpo_zh.md)
