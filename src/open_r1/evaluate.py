# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


def get_tasks():
    """Return a list of task configurations."""
    latex_gold_metric = multilingual_extractive_match_metric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(LatexExtractionConfig(),),
        # Match boxed first before trying other regexes
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
    )

    expr_gold_metric = multilingual_extractive_match_metric(
        language=Language.ENGLISH,
        fallback_mode="first_match",
        precision=5,
        gold_extraction_target=(ExprExtractionConfig(),),
        # Match boxed first before trying other regexes
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
    )

    def prompt_fn(line, task_name: str = None):
        """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
        return Doc(
            task_name=task_name,
            query=line["problem"],
            choices=[line["solution"]],
            gold_index=0,
        )

    def aime_prompt_fn(line, task_name: str = None):
        return Doc(
            task_name=task_name,
            query=line["problem"],
            choices=[line["answer"]],
            gold_index=0,
        )

    # Define tasks
    tasks = []
    
    # AIME 2024 task
    tasks.append(
        LightevalTaskConfig(
            name="aime24",
            suite=["custom"],
            prompt_function=aime_prompt_fn,
            hf_repo="HuggingFaceH4/aime_2024",
            hf_subset="default",
            hf_avail_splits=["train"],
            evaluation_splits=["train"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=32768,
            metric=[expr_gold_metric],
            version=1,
        )
    )
    
    # MATH-500 task
    tasks.append(
        LightevalTaskConfig(
            name="math_500",
            suite=["custom"],
            prompt_function=prompt_fn,
            hf_repo="HuggingFaceH4/MATH-500",
            hf_subset="default",
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=32768,
            metric=[latex_gold_metric],
            version=1,
        )
    )
    
    return tasks

# Define task registry
TASK_REGISTRY = {task.name: task for task in get_tasks()}

# Add tasks to the table
TASKS_TABLE = []
for task in get_tasks():
    TASKS_TABLE.append(task)

# Define task groups
TASK_GROUPS = {
    "custom|aime24|0|0": ["aime24"],
    "custom|math_500|0|0": ["math_500"],
}

# MODULE LOGIC
if __name__ == "__main__":
    print([t.name for t in TASKS_TABLE])
