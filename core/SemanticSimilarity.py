from sentence_transformers import SentenceTransformer
from typing import List, Union, Tuple
import numpy as np

class SemanticSimilarity():
    def __init__(self):
        self.threshold_value = 0.5
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedded_reference = []
        self.disambiguated_reference_list = []
        self.rels = []
        self.embedded_rels = []
        self.label_to_embedding = {}

        self.triplet_sentence_reference_list  = []
        self.triplet_sentence_embeddings  = []

    
    def generate_reference_without_disambiguation(self, reference_list):
        embed_ref = self._populate_embedded_reference(reference_list)
        self.disambiguated_reference_list, self.embedded_reference = reference_list, embed_ref  

    def generate_reference(self, reference_list):
        temp = self._populate_embedded_reference(reference_list)
        x, y = self._disambiguate_reference_list_entities(reference_list, temp)
        self.disambiguated_reference_list, self.embedded_reference = x, y

    def _populate_embedded_reference(self, reference_list : str) -> List[List[float]] :
        embedded_reference = []
        for i, e in enumerate(reference_list):
            embedded_reference.append(self._get_embedding_token(e))
        return embedded_reference
        
    def _get_embedding_token(self, phrase):
        return self.model.encode([phrase])[0]

    def generate_label_embedding(self, labels):
        for l in labels:
            self.label_to_embedding[l] = self._get_embedding_token(l)
    
    def generate_summary_embedding(self, summary):
        return self._get_embedding_token(summary)
        
    def _cosine(self, u, v) -> float:
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def add_entity(self,e):
        if not self.get_most_similar_entity(e):
            self.disambiguated_reference_list.append(e)
            self.embedded_reference.append(self._get_embedding_token(e))

    def add_relation(self,r):
        if not self.get_most_similar_relation(r):
            self.rels.append(r)
            self.embedded_rels.append(self._get_embedding_token(r))

    def get_most_similar_entity(self, phrase):
        return self.get_most_similar_word(phrase, task="entity")

    def get_most_similar_entities(self, phrase):
        return self.get_most_similar_word(phrase, task="entity", multiple_words=True)

    def get_most_similar_relation(self, phrase):
        return self.get_most_similar_word(phrase, task="relation")
    
        
    def get_most_similar_word(self, phrase, task="entity", multiple_words=False) -> Union[None, str]:
        query = self._get_embedding_token(phrase)
        max_score = -999
        max_index = -1

        reference, embedded_reference = [],[]
        if task=="entity":
            reference = self.disambiguated_reference_list
            embedded_reference = self.embedded_reference
        elif task=="relation":
            reference = self.rels
            embedded_reference = self.embedded_rels

        temp_res = []
        for i, ref in enumerate(embedded_reference):
            score = self._cosine(query, ref)
            if score > max_score:
                max_score = score
                max_index = i
            
            temp_res.append(score)
        
        if not multiple_words:
            if max_score > self.threshold_value:
                return reference[max_index]
        
        else:
            
            multiple_answers = []
            for i in range(len(temp_res)):
                if temp_res[i] > self.threshold_value:
                    multiple_answers.append(reference[i])
            return multiple_answers
        return None
    
    def _get_all_similarity_scores(self, embedded_reference : List):
        return np.dot(np.array(embedded_reference), np.array(embedded_reference).T)

    def _get_similar_entities(self, exclude_index : int, l : List[float]) -> List[Tuple[float,int]]:
        results = []
        for i,e in enumerate(l):
            if i != exclude_index and l[i] >= self.threshold_value:
                results.append((l[i],i))
        return results
        
    def _disambiguate_reference_list_entities(self, reference_list : List, embedded_reference_list : List) -> List[str]:
        reference_list = self.disambiguated_reference_list + reference_list
        embedded_reference_list = self.embedded_reference + embedded_reference_list
        similarity_scores = self._get_all_similarity_scores(embedded_reference_list)
        dissimilar,dissimilar_embedded, ignore = [],[],[]
        i = 0
        while i < len(similarity_scores):
            if i not in ignore:
                current = similarity_scores[i]
                max_score_indices = self._get_similar_entities(i, current)
                ignore.extend([i for v,i in max_score_indices])
                dissimilar_index = min([i] + [j for v,j in max_score_indices]) #Get the lowest index. This is because when we are parsing the second article, we want to keep any similar entities we found in the first article and remove entities any subsequent articles. E.g. ["alpha", "beta", "omega", "alpha", ""quatro, "alipha"]. If i is at 4 we just want to keep alpha so that the disambiguated list is "alpha", "beta", "omega", "quatro"]
                dissimilar.append(reference_list[dissimilar_index])
                dissimilar_embedded.append(embedded_reference_list[dissimilar_index])
                # dissimilar.append(max([reference_list[i]] + [reference_list[j] for v,j in max_score_indices])) #assuming larger text carries more information
            i += 1

        return dissimilar, dissimilar_embedded

    def _disambiguate_triplets(self, triplets):
        triplet_sentences = []
        for t in triplets:
            sen = " ".join(t) 
            
            

    
