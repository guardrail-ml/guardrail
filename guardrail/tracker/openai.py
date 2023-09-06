import time
import requests
import openai
from threading import Thread

from guardrail.metrics.utils.keys import init_openai_key, init_guardrail_key
from guardrail.metrics.utils.flatten_metrics import flatten_json
from guardrail.metrics.offline_evals.offline_metrics import OfflineMetricsEvaluator

class OpenAI:
    def __init__(self):
        init_openai_key()
        self.offline_metrics = OfflineMetricsEvaluator()

    def create_completion(self, engine, prompt, gr_tags=None, **kwargs):
        kwargs['engine'] = engine
        kwargs['prompt'] = prompt
        if gr_tags:
            kwargs['gr_tags'] = gr_tags
        result = openai.Completion.create(**kwargs)        

    def create_chat_completion(self, model, messages, **kwargs):
        kwargs['model'] = model
        kwargs['messages'] = messages

        start_time = time.time()
        openai_response = self._chat_completion_request(**kwargs)
        
        prompt = self.get_latest_user_prompt(messages)
        chatbot_response = openai_response["choices"][0]["message"]["content"]

        end_time = time.time()
        total_time = end_time - start_time
        print(f"ChatCompletion executed in {total_time:.4f} seconds")

        # Start storing logs in the background
        eval_log_thread = Thread(target=self.run_eval_store_logs, args=(prompt, chatbot_response, openai_response["usage"], total_time))
        eval_log_thread.start()

        return openai_response, eval_log_thread
     
    def get_latest_user_prompt(self, messages):
        for message in reversed(messages):
            if message["role"] == "user":
                return message["content"]
        return None

    def run_chat_completion(self, 
                            user_message_content, 
                            model="gpt-3.5-turbo", 
                            temperature=0, 
                            max_tokens=256, 
                            **kwargs):
        message = {'role': 'user', 'content': user_message_content}

        response, eval_log_thread = self.create_chat_completion(
            model=model,
            messages=[message],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        eval_log_thread.join()
        return response

    def run_chat_completion_entry(self, user_message_content, model="gpt-3.5-turbo", temperature=0, max_tokens=256, **kwargs):
        self.run_chat_completion(user_message_content, model, temperature, max_tokens, **kwargs)

    def _chat_completion_request(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)

    def run_eval_store_logs(self, prompt, output, openai_usage, total_time, model_name="gpt-3.5-turbo"):
        # Simulate storing logs
        print("prompt: ", prompt)
        print("output: ", output)
        print("OPENAI RESPONSE", openai_usage)
        metrics = self._run_offline_metrics(prompt, output)
        cost = self.calculate_openai_cost(openai_usage, model_name)

        promptEvaluations = {
            "prompt_injection": metrics["prompt_injection"],
            "relevance": metrics["relevance"],
            "toxicity": metrics["toxicity"]
        }

        outputEvaluations = {k: v for k, v in metrics.items() if k not in promptEvaluations}
        total_tokens = openai_usage["total_tokens"]

        api_key = init_guardrail_key()
        api_url = "https://guardrail-api-sknmgpkina-uc.a.run.app/v1/create_evaluation_log"
        headers = {
            'Content-Type': 'text/plain',
        }

        data = {
            'prompt': prompt,
            'output': output,
            'api_key': api_key,
            'cost': str(cost),
            'llmToken': str(total_tokens),
            'latency': str(total_time),
            'promptEvaluations': promptEvaluations,
            'outputEvaluations': outputEvaluations,
        }

        print(data)

        response = requests.post(api_url, headers=headers, json=data)
        print(response)
        if response.status_code == 200:
            print(response.json())
            return response.json()
        elif response.status_code == 400:
            print("Response:", response)
    
            # Print the response content for more details about the error
            print("Response Content:", response.content)

            # You can also print response headers for additional information
            print("Response Headers:", response.headers)
            return None
        else:
            return None


    def _run_offline_metrics(self, prompt, response):
        results = self.offline_metrics.evaluate_metrics(response, prompt)
        results = flatten_json(results)
        print(results)
        return results

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
        print(f"Total cost for {model}: ${total_cost:.6f}\n")

        return total_cost