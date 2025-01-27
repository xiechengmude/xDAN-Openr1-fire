from datasets import load_dataset
from distilabel import Pipeline
from distilabel.tasks import OpenAIChatTask
from distilabel.llm import OpenAILLM

# 定义系统提示
SYSTEM_PROMPT = """You are a helpful math assistant. When solving math problems, please:
1. Think through the problem step by step
2. Show your reasoning process
3. Provide a clear final answer
4. Use LaTeX for mathematical expressions
5. Format your response as: <think>reasoning</think><answer>final answer</answer>
"""

def build_prompt(example):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]}
    ]

def main():
    # 加载数据集
    dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))
    
    # 配置 OpenAI API
    llm = OpenAILLM(
        model="deepseek/r1-distill-qwen-14b",
        base_url="http://localhost:8001/v1",
        temperature=0.7,
        max_tokens=2048,
    )
    
    # 创建任务
    task = OpenAIChatTask(
        llm=llm,
        prompt_template=build_prompt,
        output_parser=lambda x: x["choices"][0]["message"]["content"],
    )
    
    # 创建管道
    pipeline = Pipeline(
        task=task,
        num_generations=1,
    )
    
    # 生成数据
    generated_dataset = pipeline.generate(
        dataset,
        save_path="xDAN-Distill-R1-14b-dataset.jsonl",
    )
    
    print(f"Generated {len(generated_dataset)} examples")

if __name__ == "__main__":
    main()
