import openai
from . import prompts
from ..utils.keys import init_openai_key
from ..utils.textutils import TextUtilities

class OpenAIProvider():
    model_engine: str

    def __init__(
        self, *args, model_engine="gpt-3.5-turbo", **kwargs
    ):
        self_kwargs = dict()
        self_kwargs.update(**kwargs)
        self_kwargs['model_engine'] = model_engine
        self.model_engine = model_engine
        self.text_utils = TextUtilities()

        print(init_openai_key())

    def _create_chat_completion(self, *args, **kwargs):
        return openai.ChatCompletion.create(*args, **kwargs)

    def _moderation(self, text: str):
        return openai.Moderation.create(input=text)
    
    def _run_chat_completion(self, messages: list[dict[str, str]]) -> str:
        response = self._create_chat_completion(
            model=self.model_engine,
            temperature=0.0,
            messages=messages
        )
        score = response["choices"][0]["message"]["content"]
        return response, score

    def _evaluate(self, text: str, system_prompt: str) -> float:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        response, score = self._run_chat_completion(messages)
        # openai_api_calculate_cost(response.usage, self.model_engine)

        return self.text_utils.re_1_10_score(score) / 10

    def evaluate_text(self, text: str, system_prompt: str) -> float:
        return self._evaluate(text, system_prompt)
    
    ### OpenAI Moderation
    def _calculate_inverted_score(self, category_name, text: str) -> float:
        openai_response = self._moderation(text)
        results = openai_response["results"]
        score = results[0]["category_scores"][category_name]
        return 1 - score, results

    def moderation_not_hate(self, text: str) -> float:
        return self._calculate_inverted_score("hate", text)

    def moderation_not_hatethreatening(self, text: str) -> float:
        return self._calculate_inverted_score("hate/threatening", text)

    def moderation_not_selfharm(self, text: str) -> float:
        return self._calculate_inverted_score("self-harm", text)

    def moderation_not_sexual(self, text: str) -> float:
        return self._calculate_inverted_score("sexual", text)

    def moderation_not_sexualminors(self, text: str) -> float:
        return self._calculate_inverted_score("sexual/minors", text)

    def moderation_not_violence(self, text: str) -> float:
        return self._calculate_inverted_score("violence", text)

    def moderation_not_violencegraphic(self, text: str) -> float:
        return self._calculate_inverted_score("violence/graphic", text)

    def moderation_not_harassment(self, text: str) -> float:
        return self._calculate_inverted_score("harassment", text)

    def moderation_not_harassmentthreatening(self, text: str) -> float:
        return self._calculate_inverted_score("harassment/threatening", text)

    # Define evaluation methods
    def conciseness(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_CONCISENESS_PROMPT)

    def correctness(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_CORRECTNESS_PROMPT)

    def coherence(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_COHERENCE_PROMPT)

    def harmfulness(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_HARMFULNESS_PROMPT)

    def maliciousness(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_MALICIOUSNESS_PROMPT)

    def helpfulness(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_HELPFULNESS_PROMPT)

    def controversiality(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_CONTROVERSIALITY_PROMPT)

    def misogyny(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_MISOGYNY_PROMPT)

    def criminality(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_CRIMINALITY_PROMPT)

    def insensitivity(self, text: str) -> float:
        return self.evaluate_text(text, prompts.LANGCHAIN_INSENSITIVITY_PROMPT)