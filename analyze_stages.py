import json
from pathlib import Path
from collections import defaultdict

def analyze_stage(file_path):
    """分析单个stage的数据集"""
    if not Path(file_path).exists():
        print(f"文件不存在: {file_path}")
        return
        
    samples = []
    ratings = defaultdict(int)
    detected_patterns = defaultdict(int)
    original_tags = defaultdict(int)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
            ratings[sample['difficulty']] += 1
            
            # 检测到的算法模式
            for pattern in sample.get('patterns', []):
                detected_patterns[pattern] += 1
                
            # 原始标签
            for tag in sample.get('tags', []):
                original_tags[tag] += 1
    
    print(f"\n分析文件: {file_path}")
    print(f"样本总数: {len(samples)}")
    
    print("\n难度分布:")
    for rating, count in sorted(ratings.items()):
        print(f"Rating {rating}: {count} 样本 ({count/len(samples)*100:.1f}%)")
    
    print("\n检测到的算法模式分布:")
    for pattern, count in sorted(detected_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"{pattern}: {count} 样本 ({count/len(samples)*100:.1f}%)")
        
    print("\n原始标签分布:")
    for tag, count in sorted(original_tags.items(), key=lambda x: x[1], reverse=True):
        print(f"{tag}: {count} 样本 ({count/len(samples)*100:.1f}%)")
        
    # 统计没有被识别出模式的样本
    no_pattern_samples = sum(1 for s in samples if not s.get('patterns'))
    print(f"\n未识别出模式的样本数: {no_pattern_samples} ({no_pattern_samples/len(samples)*100:.1f}%)")

def main():
    """分析所有stage的数据集"""
    data_dir = Path('processed_data/1')
    for stage in ['stage1', 'stage2', 'stage3']:
        file_path = data_dir / f'{stage}_data.jsonl'
        analyze_stage(file_path)

if __name__ == '__main__':
    main()
