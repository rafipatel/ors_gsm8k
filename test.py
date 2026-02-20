from openreward import OpenReward

or_client = OpenReward()
environment = or_client.environments.get(name="basicenvironment", base_url="http://localhost:8080")

print(environment.list_splits())

print(environment.list_tasks("train"))

print("="*50)
print(environment.list_tools())