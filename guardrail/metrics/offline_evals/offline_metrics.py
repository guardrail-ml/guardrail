import logging
from guardrail.metrics.offline_evals import Textstat, Toxicity, Relevance, PromptInjection, Sentiment, Bias

class OfflineMetricsEvaluator:
    def __init__(self):
        # Initialize the metrics classes
        self.textstat = Textstat()
        self.toxicity = Toxicity()
        self.relevance = Relevance()
        self.injection = PromptInjection()
        self.sentiment = Sentiment()
        self.bias = Bias()
        self.logger = logging.getLogger("Textstat")
        print("Initialized complete...")

    def evaluate_metrics(self, prompt, output):
        results = {}

        ts_results = self.textstat.evaluate(output)
        sentiment_results = self.sentiment.evaluate(output)
        bias_results = self.bias.evaluate(output)
        
        if prompt:
            relevance_results = self.relevance.evaluate(output, prompt)
            injection_results = self.injection.evaluate(prompt)
            toxicity_results = self.toxicity.evaluate(output, prompt)

        results["text_quality"] = ts_results
        results["toxicity"] = toxicity_results
        results["sentiment"] = sentiment_results
        results["bias"] = bias_results

        if relevance_results:
            results["relevance"] = relevance_results
        if injection_results:
            results["prompt_injection"] = injection_results

        return results
