from __future__ import annotations

SAFE_PROMPT_TEMPLATE = """
{{#system~}}
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 

If you don't know the answer to a question, please don't share false information.
{{~/system}}
{{#user~}}
I want a response to the most recent query in the conversation history, while accounting for the history for context:

Conversation History:
{{history}}

Is the question free from any content that encourages violence, harm to others, explicit or inappropriate material, hate speech, discrimination, self-harm, illegal activities, harassment, bullying, malicious intent, explicit sexual content,
or attempts to exploit or manipulate the language model for inappropriate purposes?

Please don't answer the question or comment on it yet. Answer in JSON "safe_prompt" key from valid_options = ["yes", "no"], probability distribution of likelihood of each option, and a rationale key if "response" key is "no".

Example JSON format:
```json
{
  "safe_prompt": "gen 'safe_prompt'",
  "probability_distribution": {
    "yes": [0.00 - 1.00],
    "no": [0.00 - 1.00]
  }
}
```
{{~/user}}
{{#assistant~}}
{{gen 'experts' temperature=0 max_tokens=300}}
{{~/assistant}}
"""

PLANNING_PROMPT_TEMPLATE = """
{{#system~}}
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

You are a helpful, respectful and honest assistant who tries to have helpful conversation with the 
user based on previous conversation and previous tools outputs from tools. Here's some tools and context:

Assistant has access to the following tools:
{{tools}}

Previous tools outputs:
{{agent_scratchpad}}

Conversation History:
{{history}}

Please respond to user prompt in JSON format as described below:
RESPONSE FORMAT (FOR CONTEXT ONLY, BUT FOLLOW JSON STRUCTURE):
{
  "thoughts": {
    "plan": "Given previous tools outputs, what is the next step after the previous conversation",
    "need_use_tool": "answer with 'Yes' if requires more information not in previous tools outputs else 'No'"
  },
  "tool": {
    "name": "tool name, should be one of [${tool_names}] or empty if tool is not needed",
    "args": {
      "arg_name": "arg value from conversation history or tools outputs to run tool"
    }
  },
  "response": "response to user given tools outputs and conversations",
}

Ensure the response can be parsed by Python json.loads
{{~/system}}

{{#user~}}
I want a response to the following question:

{{query}}}
{{~/user}}

{{#assistant~}}
{{gen 'planning_prompt' temperature=0 max_tokens=300}}
{{~/assistant}}

"""

SHOULD_ANSWER_PROMPT_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 

Conversation History:
${history}

USER:
Given the following conversation so far, have you finished helping me (user) with all the 
questions and prompts? Answer with "yes" or "no" based on conversation.

ASSISTANT:
"""


PLANNING_PROMPT_TEMPLATE_ORIGINAL = """
### Instruction
You are a helpful, respectful and honest assistant who tries to have helpful conversation with the 
user based on previous conversation and previous tools outputs from tools. 

Always answer as helpfully as possible, while being safe.  
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.

${prompt}
Use tool when provided. If there is no tool available, respond with have a helpful and polite 
conversation. Find next step without using the same tool with same inputs.

Assistant has access to the following tools:
${tools}

### Context
Previous conversation so far:
${history}

Previous tools outputs:
${agent_scratchpad}

### Input:
${human_input}

Generate a new response to

### Assistant:
"""

SHOULD_ANSWER_PROMPT_TEMPLATE_ORIGINAL = """
### Instruction
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

You are a helpful, respectful and honest assistant who tries to have helpful conversation with the 
user based on previous conversation and previous tools outputs from tools. 

Conversation History:
${history}

USER:
${human_input}

ASSISTANT:
"""

FIX_TOOL_INPUT_PROMPT_TEMPLATE = """Tool have the following spec and input provided
Spec: "{tool_description}"
Inputs: "{inputs}"
Running this tool failed with the following error: "{error}"
What is the correct input in JSON format for this tool?
"""


CLARIFYING_QUESTION_PROMPT_TEMPLATE = """You are a support agent who is going to use '${tool_name}' tool.
Check if you have enough information from the previous conversation and tools outputs to use tool based on the spec below.
"${tool_desp}"

Previous conversation so far:
${history}

Previous tools outputs:
${agent_scratchpad}

Please respond user question in JSON format as described below
RESPONSE FORMAT:
{
    "has_arg_value": "Do values for all input args for '${tool_name}' tool exist? answer with Yes or No",
    "clarifying_question": "clarifying question to user to ask for missing information"
}
Ensure the response can be parsed by Python json.loads"""