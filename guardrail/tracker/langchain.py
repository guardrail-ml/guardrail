import time
from typing import Any, Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult, HumanMessage
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler

from guardrail.tracker.openai import OpenAI

class GuardrailCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    def __init__(self):
        super().__init__()
        self.prompts = []  # Initialize an empty list to store prompts
        self.openai_tracker = OpenAI()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("Guardrail is tracking...")
        print(prompts)

        # Store prompts in the instance variable
        self.prompts = prompts

        # Record the start time
        self.start_time = time.time()

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("LLM chain is ending...")
        print(response)

        # Calculate the total time elapsed
        end_time = time.time()
        total_time = end_time - self.start_time
        print(f"Total time elapsed: {total_time} seconds")

        prompt = self.prompts[-1]
        chatbot_response = response.generations[0][0].text
        token_usage = response.llm_output["token_usage"]

        self.openai_tracker.run_eval_store_logs(prompt, chatbot_response, token_usage, total_time)
