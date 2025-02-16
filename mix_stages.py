import json
import yaml
import random
from pathlib import Path
from collections import defaultdict

def load_config():
    """加载配置文件"""
    with open('configs/code_process_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_stage_data(stage_file):
    """加载单个stage的数据"""
    samples = []
    with open(stage_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def analyze_mixed_data(samples):
    """分析混合数据集的分布"""
    stage_counts = defaultdict(int)
    pattern_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    for sample in samples:
        stage_counts[f"stage{sample['stage']}"] += 1
        for pattern in sample.get('patterns', []):
            pattern_counts[pattern] += 1
        for tag in sample.get('tags', []):
            tag_counts[tag] += 1
    
    total = len(samples)
    print(f"\n混合数据集总样本数: {total}")
    
    print("\n各阶段分布:")
    for stage, count in sorted(stage_counts.items()):
        print(f"{stage}: {count} 样本 ({count/total*100:.1f}%)")
    
    print("\n算法模式分布:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{pattern}: {count} 样本 ({count/total*100:.1f}%)")
    
    print("\n原始标签分布:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{tag}: {count} 样本 ({count/total*100:.1f}%)")

def create_mixed_dataset(config):
    """创建混合数据集"""
    data_dir = Path('processed_data/1')
    output_dir = data_dir / 'mixed'
    output_dir.mkdir(exist_ok=True)
    
    # 加载各阶段数据
    stage_data = {}
    for i in range(1, 4):
        stage_file = data_dir / f'stage{i}_data.jsonl'
        stage_data[i] = load_stage_data(stage_file)
    
    # 获取配置中的混合比例
    mix_ratios = config['mixed_data_ratios']
    total_samples = config['mixed_data_total_samples']
    
    # 计算每个阶段需要的样本数
    stage_samples = {
        stage: int(ratio * total_samples)
        for stage, ratio in mix_ratios.items()
    }
    
    # 调整样本数以确保总和等于目标数量
    total = sum(stage_samples.values())
    if total < total_samples:
        # 将剩余的样本添加到stage1
        stage_samples['stage1'] += total_samples - total
    
    # 随机选择样本并添加stage标记
    mixed_samples = []
    for stage, count in stage_samples.items():
        stage_num = int(stage[-1])  # 从'stage1'提取数字1
        samples = random.sample(stage_data[stage_num], min(count, len(stage_data[stage_num])))
        for sample in samples:
            sample['stage'] = stage_num
        mixed_samples.extend(samples)
    
    # 打乱样本顺序
    random.shuffle(mixed_samples)
    
    # 保存混合数据集
    output_file = output_dir / 'mixed_data.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in mixed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"混合数据集已保存到: {output_file}")
    return mixed_samples

def main():
    """主函数"""
    config = load_config()
    mixed_samples = create_mixed_dataset(config)
    analyze_mixed_data(mixed_samples)

if __name__ == '__main__':
    main()
