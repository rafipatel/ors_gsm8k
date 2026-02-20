from openai import OpenAI
from openreward import OpenReward
import json

or_client = OpenReward()
oai_client = OpenAI()
MODEL_NAME = "gpt-5.2"

environment = or_client.environments.get(name="gsm8k", base_url="http://localhost:8080")
tasks = environment.list_tasks(split="train")
tools = environment.list_tools(format="openai")

print(f"\n{'='*60}")
print(f"ENVIRONMENT SETUP")
print(f"{'='*60}")
print(f"Total tasks loaded: {len(tasks)}")
print(f"{'='*60}\n")

example_task = tasks[0]
print(f"Task content: {example_task}\n")

with environment.session(task=example_task) as session:
    prompt = session.get_prompt()
    input_list = [{"role": "user", "content": prompt[0].text}]
    finished = False
    
    print(f"\n{'='*60}")
    print(f"INITIAL PROMPT")


    step = 0
    while not finished:
        step += 1
        print(f"\n{'='*60}")
        print(f"STEP {step}: Calling LLM")
        print(f"{'='*60}")
        print(input_list)
        print(f"{'='*60}\n")
        response = oai_client.responses.create(
            model=MODEL_NAME,
            tools=tools,
            input=input_list
        )
        
        print(f"Response output: {response.output}")
        print(f"Output types: {[item.type for item in response.output]}")

        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                print(f"\n{'='*60}")
                print(f"TOOL CALL DETECTED")
                print(f"{'='*60}")
                print(f"Tool name: {item.name}")
                print(f"Arguments: {item.arguments}")
                print(f"Call ID: {item.call_id}")
                
                tool_result = session.call_tool(item.name, json.loads(str(item.arguments)))

                reward = tool_result.reward
                finished = tool_result.finished

                print(f"\nTOOL RESULT:")
                print(f"  Reward: {reward}")
                print(f"  Finished: {finished}")
                print(f"  Response: {tool_result.blocks[0].text if tool_result.blocks else 'No response'}")
                print(f"{'='*60}")

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({
                        "result": tool_result.blocks[0].text
                    })
                })

                print(f"\nFunction output added to conversation: {input_list[-1]}")

                if tool_result.finished:
                    finished = True
                    print(f"\n{'*'*60}")
                    print(f"EPISODE FINISHED")
                    print(f"Final Reward: {reward}")
                    print(f"Total Steps: {step}")
                    print(f"{'*'*60}\n")
                    break

print("\n✅ Session complete!")