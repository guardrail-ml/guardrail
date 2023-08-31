from sentence_transformers import SentenceTransformer, util

# Used for evaluating model on training / test datasets
class Similarity:
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)

    def evaluate(self, response, answer, correct_threshold=0.3):
        response_embedding = self.embedder.encode(response, convert_to_tensor=True)
        correct_answer_embedding = self.embedder.encode(answer, convert_to_tensor=True)
        
        cosine_similarity = util.pytorch_cos_sim(response_embedding, correct_answer_embedding).item()
        print('Cosine similarity is ' + str(cosine_similarity))

        # Adjust the threshold as needed, >30% threshold
        is_correct = cosine_similarity >= correct_threshold 

        return is_correct, cosine_similarity
