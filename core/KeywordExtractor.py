from keybert import KeyBERT
import nltk
from transformers import pipeline 
import re

class KeywordExtractor:
    def __init__(self, model_name="eventdata-utd/conflibert-named-entity-recognition"):
        self.pipe = pipeline("token-classification", model=model_name, tokenizer=model_name)
        self.kw_model = KeyBERT()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.semantic_matcher = None
        
    def sanitize_and_repair(func):
        def wrapper(self, *args, **kwargs):
            phrases = func(self, *args, **kwargs)
            for i, phrase in enumerate(phrases):
                phrase = phrase.strip()
                phrase = phrase.lower()
                phrase = re.sub(r'[,\.!?]', ' ', phrase)
                phrase = re.sub(r'[^a-z0-9\s]', ' ', phrase)
                phrase = re.sub(r"^\s*(?:the|a)\s+", "", phrase, flags=re.IGNORECASE)
                phrase = re.sub(r'\s+', ' ', phrase)
                phrases[i] = phrase
            return phrases
        return wrapper

    def remove_stopwords(func):
        def wrapper(self, *args, **kwargs):
            phrases = func(self, *args, **kwargs)
            filtered_words = []
            for phrase in phrases:
                if phrase.lower() not in self.stopwords:
                    filtered_words.append(" ".join([token for token in phrase.split() if token.lower() not in self.stopwords]))
                   
            return filtered_words
        return wrapper

    # @remove_stopwords 
    # @sanitize_and_repair
    def extract_named_entities(self, doc):
        results = self.pipe(doc)
        named_entities = []
        i = 0
        while i < len(results):
            result = results[i]
            if result["entity"] != "O" and result["entity"].split("-")[1] not in ["Quantity","Temporal","Money"]:
                if "B-" in result["entity"]:
                    j = i + 1
                    while j < len(results):
                        if "B-" in results[j]["entity"]:
                            break
                        j +=1
                    
                    full_word = []
                    for x in range(i,j):
                        w = results[x]["word"]
                        if "##" in w:
                            if full_word:
                                full_word[-1] = full_word[-1] + re.sub(r'[#]', '', w)
                            else:
                                full_word.append(re.sub(r'[#]', '', w))
                        else:
                            full_word.append(w)
                    
                    named_entities.append(" ".join(full_word))
                    i = j-1
            i += 1
        return named_entities
        
   

    # @sanitize_and_repair
    # @remove_stopwords
    def extract_keywords(self, doc):
        return [k[0] for k in self.kw_model.extract_keywords(doc)]
    
    
    