from distilabel.models.llms import OpenAILLM
from distilabel.steps.tasks import ChatGeneration
from distilabel.pipeline import Pipeline
from datasets import load_dataset

SYSTEM_PROMPT = """You are a helpful math assistant. When solving math problems, please:
1. Think through the problem step by step
2. Show your work clearly
3. Double check your answer
"""

def get_messages(example):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]}
    ]

def main():
    # 加载数据集
    dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

    # 配置 OpenAI API
    llm = OpenAILLM(
        model="xDAN-L2-Reasoner",
        base_url="http://0.0.0.0:8001/v1",
        generation_kwargs={
            "temperature": 0.6,
            "max_tokens": 2048
        }
    )

    # 创建任务
    task = ChatGeneration(
        llm=llm,
        messages=get_messages
    )

    # 创建并运行 pipeline
    with Pipeline() as pipeline:
        pipeline.add_task(task)
        generated_dataset = pipeline.run(dataset=dataset)

    print(f"Generated {len(generated_dataset)} examples")

if __name__ == "__main__":
    main()