from math_verify import parse, verify
import pandas as pd
from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

class GSM8KTaskSpec(BaseModel):
    id: str
    question: str
    answer: str


class AnswerParams(BaseModel):
    answer: str


train_tasks = pd.read_parquet("/orwd_data/train-00000-of-00001.parquet").to_dict(orient="records")
test_tasks = pd.read_parquet("/orwd_data/test-00000-of-00001.parquet").to_dict(orient="records")

for i, task in enumerate(train_tasks):
    task['id'] = str(i)
for i, task in enumerate(test_tasks):
    task['id'] = str(i)


class GSM8K(Environment):
    """
    A GSM8K environment
    """
    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = GSM8KTaskSpec.model_validate(task_spec)

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            return train_tasks
        elif split == "test":
            return test_tasks
        raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    def get_prompt(self) -> str:
        return [TextBlock(type="text", text=self.config.question)]

    @tool
    def answer(self, params: AnswerParams) -> ToolOutput:
        """
        The answer tool can be used to submit your final answer. Note that this finishes the episode.
        """
        gold = parse(self.config.answer)
        print("="*50)
        print(gold)
        print("="*50)
        answer = parse(params.answer)
        print("="*50)
        print(answer)
        print("="*50)
        is_correct = verify(gold, answer)

        if is_correct:
            agent_message = "Correct!"
            reward = 1.0
        else:
            agent_message = "Wrong!"
            reward = 0.0

        return ToolOutput(
            blocks=[TextBlock(type="text", text=agent_message)],
            reward=reward,
            finished=True
        )

if __name__ == "__main__":
    Server([GSM8K]).run()