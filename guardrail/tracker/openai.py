import threading
import time
import openai
from guardrail.metrics.utils.keys import init_openai_key

class OpenAIWrapper:
    def __init__(self):
        init_openai_key()

    def create_completion(self, engine, prompt, gr_tags=None, **kwargs):
        kwargs['engine'] = engine
        kwargs['prompt'] = prompt
        if gr_tags:
            kwargs['gr_tags'] = gr_tags
        result = openai.Completion.create(**kwargs)

    def create_chat_completion(self, model, messages, gr_tags=None, **kwargs):
        kwargs['model'] = model
        kwargs['messages'] = messages

        if gr_tags:
            kwargs['gr_tags'] = gr_tags

        start_time = time.time()
        response = self._chat_completion_request(**kwargs)
        end_time = time.time()
        print(f"ChatCompletion executed in {end_time - start_time:.4f} seconds")
        self.calculate_openai_cost(response.usage, "gpt-3.5-turbo")
        return response["choices"][0]["message"]["content"]

    def _chat_completion_request(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def calculate_openai_cost(self, usage, model="gpt-3.5-turbo"):
        pricing = {
            'gpt-3.5-turbo': {
                'prompt': 0.0015,
                'completion': 0.002,
            },
            'gpt-3.5-turbo-16k': {
                'prompt': 0.003,
                'completion': 0.004,
            },
            'gpt-4-8k': {
                'prompt': 0.03,
                'completion': 0.06,
            },
            'gpt-4-32k': {
                'prompt': 0.06,
                'completion': 0.12,
            },
            'text-embedding-ada-002-v2': {
                'prompt': 0.0001,
                'completion': 0.0001,
            }
        }

        try:
            model_pricing = pricing[model]
        except KeyError:
            raise ValueError("Invalid model specified")

        prompt_cost = usage['prompt_tokens'] * model_pricing['prompt'] / 1000
        completion_cost = usage['completion_tokens'] * model_pricing['completion'] / 1000

        total_cost = prompt_cost + completion_cost
        print(f"\nTokens used:  {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion = {usage['total_tokens']:,} tokens")
        print(f"Total cost for {model}: ${total_cost:.4f}\n")

        return total_cost
