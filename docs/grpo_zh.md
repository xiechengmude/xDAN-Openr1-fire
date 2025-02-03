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

def format_reward(completions):
    # 验证输出格式是否符合规范
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
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

3. **超参数调优**：
   - 调整KL惩罚系数(β)
   - 优化组大小设置
   - 平衡不同奖励组件

## 使用示例

```bash
python -m src.open_r1.grpo \
    --model_name_or_path deepseek-ai/deepseek-base \
    --dataset_name math_dataset \
    --reward_funcs accuracy format \
    --output_dir output/grpo_math
```
