import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache
from openai import AsyncOpenAI
from contextlib import asynccontextmanager

class LLMRewardConfig:
    """LLM reward configuration"""
    def __init__(
        self,
        api_base: str = "http://localhost:8001/v1",
        model: str = "deepseek/r1-distill-qwen-32b-202407",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ):
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

class AsyncLLMReward:
    """Asynchronous LLM-based reward calculator"""
    
    def __init__(self, config: LLMRewardConfig):
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.api_base,
            timeout=config.timeout,
        )
        self._load_reward_criteria()
        
    def _load_reward_criteria(self):
        """Load reward criteria from config file"""
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "reward_criteria.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.reward_criteria = config['llm_reward_criteria']['dimensions']
        except Exception as e:
            print(f"Warning: Failed to load reward criteria config: {str(e)}")
            # 使用默认配置
            self.reward_criteria = [
                {"name": "accuracy", "weight": 0.4, "description": "答案的准确性 - 回答是否与参考答案在关键信息和结论上一致"},
                {"name": "logic", "weight": 0.3, "description": "逻辑性和连贯性 - 论述是否清晰、连贯，推理过程是否合理"},
                {"name": "completeness", "weight": 0.2, "description": "完整性 - 是否涵盖了问题的所有重要方面"},
                {"name": "clarity", "weight": 0.1, "description": "表达清晰度 - 语言是否清晰、专业、易懂"}
            ]
        
    @asynccontextmanager
    async def session(self):
        try:
            yield self
        finally:
            await self._client.close()

    @staticmethod
    def _create_eval_prompt(completion: str, solution: str, reward_criteria) -> str:
        """Create evaluation prompt for the LLM"""
        criteria_text = "\n".join(
            f"{i+1}. {dim['description']} ({int(dim['weight']*100)}%)"
            for i, dim in enumerate(reward_criteria)
        )
        
        return f"""请作为一个专业的评估专家，评估以下回答的质量。评分标准：
{criteria_text}

参考答案：
{solution}

待评估回答：
{completion}

请直接输出一个0到1之间的分数（如0.85），表示综合评分。不要输出任何其他内容。"""

    async def _call_llm_api(self, prompt: str) -> float:
        """Call LLM API and get the score"""
        try:
            response = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            return 0.0

    @lru_cache(maxsize=1024)
    def _get_cached_score(self, completion_key: str, solution_key: str) -> Optional[float]:
        """Get cached score if available"""
        return None  # Implement actual caching if needed

    def _cache_score(self, completion_key: str, solution_key: str, score: float):
        """Cache the score"""
        pass  # Implement actual caching if needed

    async def calculate_rewards(self, completions: List[Dict[str, Any]], solution: str) -> List[float]:
        """Calculate rewards for a batch of completions"""
        rewards = []
        for completion in completions:
            content = completion[0]["content"]
            # Check cache first
            cache_key = (content, solution)
            cached_score = self._get_cached_score(*cache_key)
            
            if cached_score is not None:
                rewards.append(cached_score)
                continue
            
            # Create evaluation prompt
            prompt = self._create_eval_prompt(content, solution, self.reward_criteria)
            
            # Get score from LLM
            score = await self._call_llm_api(prompt)
            
            # Cache the result
            self._cache_score(*cache_key, score)
            rewards.append(score)
        
        return rewards

async def _calculate_rewards_async(completions: List[Dict[str, Any]], solution: str, config: LLMRewardConfig) -> List[float]:
    """Async helper function for reward calculation"""
    async with AsyncLLMReward(config).session() as reward_calculator:
        return await reward_calculator.calculate_rewards(completions, solution)

def llm_reward(completions: List[Dict[str, Any]], solution: str, **kwargs) -> List[float]:
    """LLM-based reward function for GRPO"""
    import asyncio
    
    # Create config from environment variables or defaults
    config = LLMRewardConfig(
        api_base=os.getenv("LLM_API_BASE", "http://localhost:8001/v1"),
        model=os.getenv("LLM_MODEL", "deepseek/r1-distill-qwen-32b-202407"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        timeout=float(os.getenv("LLM_TIMEOUT", "60.0")),
    )
    
    try:
        return asyncio.run(_calculate_rewards_async(completions, solution, config))
    except Exception as e:
        print(f"Error calculating LLM rewards: {str(e)}")
        return [0.0] * len(completions)
