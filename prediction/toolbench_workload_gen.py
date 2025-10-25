import json
import csv
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import random

random.seed(42)

with open('../toolllama_G123_dfs_train_new.json', 'r') as file:
    data = json.load(file)

# Define the tokenizer
model_name_or_path = "facebook/opt-125m"
padding_side = "left" if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")) else "right"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)

# Lists to hold the filtered data
prompts = []
completions = []
labels = []
prompt_lens = []

print(len(data))

s = 0

err = 0

pattern = r'for tool \"([^\"]+)\"'

def get_time(filepath='../api_response_data.csv'):
    # Initialize an empty list to store the dictionaries
    time_dict = {}
    with open(filepath, mode='r') as file:
        csv_reader = csv.DictReader(file)
        # Iterate over each row in the CSV
        for row in csv_reader:
            name = row["API"]
            name = name.replace("-", "_")
            response = {
                "time": int(row["Delay"]) / 1000.0,
            }
            time_dict[name] = response

    # Now responses_list contains all the dictionaries
    return time_dict

time_dict = get_time()

def assistant_output_parser(string):
    thought = [string[string.find("Thought: ") + len("Thought: "): string.find("\nAction: ")]]
    action = [string[string.find("Action: ") + len("Action: "): string.find("\nAction Input: ")]]
    action_input = [string[string.find("Action Input: ") + len("Action Input: "):]]
    try:
        arguments = eval(action_input[0])
    except SyntaxError:
        arguments = {}
    except Exception:
        arguments = {}
    message = {
        "role": "assistant",
        "content": thought[0],
        "function_call": {
            "name": action[0],
            "arguments": arguments
        }
    }
    return message

# Group conversations to ensure turns from the same conversation stay together
grouped_conversations = []
for idx, req in tqdm(enumerate(data)):
    items = req['conversations']
    conversation_prompts = []
    conversation_completions = []
    conversation_labels = []
    conversation_api_times = []
    conversation_api_names = []
    conversation_api_tokens = []
    tool_time_dicts = []
    times = {}
    api_token = ''
    p = ''
    legal = True
    user_cnt = 0
    for i, item in enumerate(items):
        if item['from'] == 'system':
            token_prompt = tokenizer(item['value'], truncation=True, padding=False)
            prompt_length = len(token_prompt['input_ids'])
            if prompt_length > 2048:
                legal = False
                break
            system_prompt = item['value']
            functions = system_prompt.split('Specifically, you have access to the following APIs: ')[1]
            functions = eval(functions)
            tools = []
            # check if functions in the response list
            for function in functions:
                if function.get('name') == 'Finish':
                    continue
                # Search for the pattern in the text
                match = re.search(pattern,function.get('description'))
                # Extract the matched substring if a match is found
                parent_tool_name = match.group(1) if match else None
                function["parent_tool"] = parent_tool_name
                tools.append(parent_tool_name)
            tools = list(set(tools))
            for function in functions:
                if function.get('name') == 'Finish':
                    continue
                # check if all tools in the response list
                if function["parent_tool"] not in time_dict:
                    legal = False
                    break
                else:
                    times[function["name"]] = time_dict[function["parent_tool"]]['time']
        elif item['from'] == 'user':
            p += item['value'].lstrip()
            user_cnt += 1
            if user_cnt > 1:
                err += 1
                legal = False
                break
        elif item['from'] == 'assistant':
            # Tokenize the completion and calculate its length
            tokenized_completion = tokenizer(item['value'], truncation=True, padding=False)
            token_length = len(tokenized_completion['input_ids'])
            token_prompt = tokenizer(p, truncation=True, padding=False)
            prompt_length = len(token_prompt['input_ids'])
            if prompt_length > 2048:
                # exceeded the max token length
                break

            # Store the turns
            if len(conversation_completions) > 0:
                conversation_api_tokens.append(api_token.encode('utf-8', 'replace').decode())
                api_token = ''
            conversation_prompts.append(p.encode('utf-8', 'replace').decode())
            conversation_completions.append(item['value'].encode('utf-8', 'replace').decode())
            conversation_labels.append(token_length)

            parsed_message = assistant_output_parser(item['value'])
            function_name = parsed_message["function_call"]["name"]
            if function_name != "Finish":
                if function_name not in times:
                    legal = False
                    break
                else:
                    conversation_api_times.append(times[function_name])
                    conversation_api_names.append(function_name.encode('utf-8', 'replace').decode())
            else:
                conversation_api_times.append(0)
                conversation_api_names.append("Finish")
            tool_time_dicts.append(times)

            if i != (len(items) - 1):
                p += item['value']
        elif item['from'] == 'function':
            # try to parse the API response
            try:
                function_prompt = eval(item.get('value'))
            except:
                try:
                    function_prompt = eval(item.get('value') + '"}')
                except:
                    start = 1
                    while True:
                        # use recurrence to parse unfinished utf-8 string
                        try:
                            function_prompt = eval(item.get('value')[:-start] + '"}')
                            break
                        except:
                            start += 1
                            if start > 10:
                                print("Parse failed")
                                function_prompt = {"response": ""}
                                legal = False
                                break
            if not legal:
                break
            if function_prompt['error'] != '':
                legal = False
                break
            p += f"Response: {function_prompt['response']}"
            api_token = function_prompt['response']
    
    if conversation_prompts and conversation_completions and conversation_labels and legal:
        conversation_api_tokens.append(api_token.encode('utf-8', 'replace').decode())
        grouped_conversations.append({
            'prompts': conversation_prompts,
            'completions': conversation_completions,
            'labels': conversation_labels,
            'api_times': conversation_api_times,
            'api_names': conversation_api_names,
            'api_tokens': conversation_api_tokens,
            'tool_time_dicts': tool_time_dicts
        })

# Shuffle the grouped conversations
random.shuffle(grouped_conversations)

# Split into train and validation sets
split_ratio = 0.8
train_size = int(len(grouped_conversations) * split_ratio)

train_conversations = grouped_conversations[:train_size]
valid_conversations = grouped_conversations[train_size:]

def convert_conversations_to_infercept_format(conversations):
    dataset = {}
    for i, conv in enumerate(conversations):
        dataset[f"{i}"] = []
        for j in range(len(conv['prompts'])):
            if conv['api_times'][j] == 0:
                dataset[f"{i}"].append({
                    "prompt": conv['prompts'][j],
                    "completion": conv['completions'][j],
                    "completion_length": conv['labels'][j],
                })
            else:
                dataset[f"{i}"].append({
                    "prompt": conv['prompts'][j],
                    "completion": conv['completions'][j],
                    "completion_length": conv['labels'][j],
                    "api_token": conv['api_tokens'][j],
                    "api_time": conv['api_times'][j],
                    "api_name": conv['api_names'][j],
                    "tool_time_dict": conv['tool_time_dicts'][j]
                })
    return dataset

# Convert the conversations to the Infercept format
# train_infercept_format = convert_conversations_to_infercept_format(train_conversations)
valid_infercept_format = convert_conversations_to_infercept_format(valid_conversations)

# # Save the Infercept format to a JSON file
# with open('toolbench_train_multi_fixed_full_train_w_api_name_copy.json', 'w') as file:
#     json.dump(train_infercept_format, file)

with open('toolbench_train_multi_fixed_full_val_w_api_name_copy.json', 'w') as file:
    json.dump(valid_infercept_format, file)