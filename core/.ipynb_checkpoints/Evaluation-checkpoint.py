import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

class Evaluation:
    def __init__(self):
        self.threshold = 0.5
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def _cosine(self, u, v) -> float:
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def _get_embedding(self, text):
        return self.embedding_model.encode([text])[0]
            
    def check_answer_semantically_similar(self, answer, label):
        answer_embedding, label_embedding = self._get_embedding(answer), self._get_embedding(label)
        score = self._cosine(answer_embedding, label_embedding)
        if score >= self.threshold:
            return True
        return False
    
    def check_if_label_exists_in_answer(self, answer, label):
        words = answer.split()
        for word in words:
            similarity = fuzz.ratio(word, label)
            if similarity >= self.threshold:
                return True
        return False


    
    
        
        