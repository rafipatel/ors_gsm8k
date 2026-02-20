
from openreward import OpenReward


# or_client = OpenReward()
or_client = OpenReward(base_url="http://localhost:8080") # you must point towards the environment if running locally

# Connect to local server (no namespace = localhost)
environment = or_client.environments.get(name="basicenvironment")
tasks = environment.list_tasks(split="train")
example_task = tasks[0]

with environment.session(task=example_task) as session:
    prompt = session.get_prompt()
    tool_result = session.call_tool("answer", {"answer": "4x"})
    print(prompt)
    print(tool_result)