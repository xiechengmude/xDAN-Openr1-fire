# 多阶段训练流程

DeepSeek-R1通过精心设计的四阶段训练流程实现卓越性能：

## 阶段概览
1. 监督微调(SFT)
2. 推理强化学习(RL)
3. 拒绝采样与SFT
4. 实用强化学习(RL)

## 阶段1：监督微调(SFT)

### 目标
- 建立基础推理能力
- 提升输出连贯性
- 为RL阶段奠定基础

### 实现示例
```python
sft_config = {
    "model_name": "deepseek-base",
    "dataset": "cot_dataset",
    "max_seq_length": 10000,
    "learning_rate": 2e-5
}
```

## 阶段2：推理强化学习

### 核心组件
1. 基于规则的准确率奖励
2. 推理格式强制规范
3. 语言一致性奖励

## 阶段3：拒绝采样与SFT

### 数据生成
1. 生成60万推理样本
2. 创建20万通用样本
3. 严格的质量过滤

## 阶段4：实用强化学习

### 奖励组合
```python
def compute_final_reward(completion):
    return (
        0.4 * 推理奖励 +
        0.3 * 实用奖励 +
        0.3 * 安全奖励
    )
```

## 训练建议
1. **数据质量**：
   - 严格筛选训练数据
   - 保持任务多样性
2. **稳定性控制**：
   - 实施梯度裁剪
   - 设置早停机制
3. **评估体系**：
   - 定期基准测试
   - 跨任务性能检查

## 完整训练流程
```bash
make sft-train       # 阶段1
make rl-train-reasoning  # 阶段2
make generate-synthetic   # 阶段3数据生成
make sft-train-expanded   # 阶段3训练
make rl-train-utility    # 阶段4
```
