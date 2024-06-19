import spacy
from SemanticSimilarity import SemanticSimilarity
from KeywordExtractor import KeywordExtractor
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama
import re

class Search:
    def __init__(self, s : SemanticSimilarity, kwe : KeywordExtractor, graph : Graph):
        self.s = s
        self.kwe = kwe
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.graph = graph
        self.threshold = 0.5
        self.window_size = 500        

    def _cosine(self, u, v) -> float:
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def _get_embedding(self, text):
        return self.embedding_model.encode([text])[0]
        
    def _retrieve_keywords(self, question):
        return self.kwe.extract_keywords(question)
    # def get_node_relations(self, node):
        
    def _retrieve_verbs(self, question):
        doc = self.nlp(question)
        return [token.lemma_ for token in doc if token.pos_ == "VERB"]

    def _get_most_relevant_relation_index(self, query_verb_embeddings, retrieved_node_relation):
        retrieved_node_relation_embedding = self._get_embedding(retrieved_node_relation)
        scores = self._cosine(query_verb_embeddings, retrieved_node_relation_embedding)
        max_element = max(scores)
        # print(f"score is {max_element}")
        max_index = None
        if max_element > self.threshold:
            max_index = np.argmax(scores)
            return max_index
        return None

    def _text_is_relevant(self, query_text, relation_text):
        if relation_text:
            query_text_embedding, relation_text_embedding = self._get_embedding(query_text), self._get_embedding(relation_text)
            return self._cosine(query_text_embedding, relation_text_embedding) > self.threshold
        return False
        
    def _get_relevant_texts(self, query, keywords, question_verb_phrases):
        all_texts = []
        # query_verb_embeddings = None
        # if question_verb_phrases:
        #     query_verb_embeddings = [self._get_embedding(p) for p in question_verb_phrases]
            
        # print(f"the keywords extracted are {keywords}")
        # for k in keywords:
        #     graph_entities = self.s.get_most_similar_entities(k)
            
        #     print(f'the most similar entities available in the graph to the keyword {k} are ')
        #     print(graph_entities)
        #     for graph_entity in graph_entities:
                
        #         graph_entity = re.sub(r'[^a-zA-Z]', '', graph_entity)
        #         graph_entity = re.sub(r' ', '_', graph_entity)
        #         if graph_entity == "":
        #             continue
        #         d = self.graph.run(f"MATCH (n:{graph_entity}) RETURN n").data()
        #         if d:
        #             print(f"{graph_entity} exists in the graph")
        #             nodes = self.graph.run(f"MATCH p = (n:{graph_entity})-[r]-(m) WHERE type(r) <> 'filepath' RETURN p,r['text'] as text, type(r) as relationType ").data()
        #             prev_all_texts_size =len(all_texts)
        #             for n in nodes:
        #                 print(f"processing relation {n['relationType']} of {graph_entity}")
        #                 if query_verb_embeddings: #if verbs exist in the query, get the relations which are most relevant to the verbs in the query. If no verbs are available just take in all the texts associated with the connected entities 
        #                     relevant_relation_index = self._get_most_relevant_relation_index(query_verb_embeddings, n['relationType'])
        #                     if relevant_relation_index is not None: #if relevant relation available, get associated text
        #                         relevant_relation = question_verb_phrases[relevant_relation_index]
        #                         print(f"Found a relevant relation type")
        #                         print(relevant_relation)
        #                         if self._text_is_relevant(query, n["text"]):
        #                             print('Associated text')
        #                             print(n["text"])
        #                             all_texts.append(n["text"])
                        
        #                         # print('Associated text')
        #                         # print(n["text"])
        #                         # all_texts.append(n["text"])
                    
        #                 else:    #If no verbs extracted from the question
        #                     print("no relevant relation was found")
        #                     all_texts.append(n["text"])
                    
                    # if len(all_texts) == prev_all_texts_size: #If no text was added, just get all the text from the original source text where the entity was found.
                    #     print('No relation text was found. Getting the text of the graph entity')
                    #     nodes = self.graph.run(f"MATCH p = (n:{graph_entity})-[r]-(m) WHERE type(r) = 'filepath' RETURN p,r['text'] as text, type(r) as relationType ").data()
                    #     for n in nodes:
                    #         all_texts.append(n["text"])
                        
            
                    # text = graph.run(f"MATCH p = (n:{graph_entity})-[r:filepath]->(m) RETURN r['text'] as text ").data()
        if len(all_texts) == 0:
            already_found = []
            print('these are the keywords')
            print(keywords)
            for k in keywords:
                
                graph_entity = self.s.get_most_similar_entity(k)
                if graph_entity:
                    print('the graph entity is')
                    print(graph_entity)
                    if graph_entity != "":
                        nodes = self.graph.run(f"MATCH p = (n:`{graph_entity}`)-[r]-(m) WHERE type(r) = 'filepath' RETURN p,r['text'] as text, type(r) as relationType ").data()
                        for n in nodes:
                            if n['text'] not in already_found:
                                all_texts.append(n["text"])
                                already_found.append(n['text'])
            
        return all_texts

    def _generate_query_label_nodes(self, query):
        prompt = f"""Task:Given the text {query}, generate some labels which best classify the theme of the text"""
        labels = ollama.generate(model="mistral", prompt=prompt)["response"]
        pattern = r'\d+\.\s*(.+)'
        matches = re.findall(pattern, labels)
        doc_triplets = []
        all_labels = [m for m in matches]
        doc_triplets.extend(all_labels)
        return doc_triplets

    def find_relevant_segment_from_text(self, question, all_texts):
        i = 0
        relevant_segments = []
        for text in all_texts:
            while i < len(text):
                splitted_text = text[i:i+self.window_size]
                print("here is splitted text")
                print(splitted_text)
                answer = ollama.generate('mistral', prompt=f"Given the question: {question}, reply only with 'yes' if the answer is in the text, else reply with 'no' \n ###Text: {splitted_text}")['response']
                if "yes" in answer or "Yes" in answer:
                    print("here is a relevant segment")
                    print(splitted_text)
                    relevant_segments.append(splitted_text)
                i += self.window_size
        return relevant_segments
        
    def get_label_relevant_entities(self, query):
        labels = self._generate_query_label_nodes(query)
        x = []
        for l in labels:
            x.extend(self.s.get_most_similar_entities(l))
        return list(set(x))
        
    def get_spliced_triplets(self, entity):
        nodes = self.graph.run(f"MATCH p=(n:`{entity}`)-[r]-(m) RETURN n['id'] as source,type(r) as relation,m['id'] as object LIMIT 25")
        return [n["source"] + " " + n["relation"] + " " + n["object"] for n in nodes]

    def get_all_connected_entities(self, entity):
        nodes = self.graph.run(f"MATCH p=(n:`{entity}`)-[r]-(m) RETURN m['id'] as object LIMIT 25").data()
        return [n["object"] for n in nodes]
               
    def get_all_relations(self, entity):
        nodes = self.graph.run(f"MATCH p=(n:`{entity}`)-[r]-(m) RETURN type(r) as relation LIMIT 25").data()
        return [n["relation"] for n in nodes]

    #main function to retrieve relations
    def get_label_relevant_triplets(self, query):
        labels = self.get_label_relevant_entities(query)
        print(labels)
        relations = []
        for l in labels:
            relations.extend(self.get_spliced_triplets(l))
        return list(set(relations))
    
    def retrieve_summary(self, label):
        nodes = self.graph.run(f"MATCH (n:`{label}`)-[r:category]->(m) RETURN n['id'] as source, m['id'] as object").data()
        return [n["object"] for n in nodes]

    
    def get_answers(self, question):
        keywords = self.kwe.extract_named_entities(question)
        # keywords = self._retrieve_keywords(question)
        question_verb_phrases = self._retrieve_verbs(question)
        print(f"The verbs in the question are {question_verb_phrases}")
        all_texts = self._get_relevant_texts(question, keywords, question_verb_phrases)
        all_texts = [i for i in all_texts if i]
        print(all_texts)
        relevant_text_segments = self.find_relevant_segment_from_text(question, all_texts)
        if relevant_text_segments:
            all_texts =  "\n".join(relevant_text_segments)
            return ollama.generate('mistral', prompt=f"Given the question: {question}, find the answer from the following text. \n ###Text: {all_texts}")['response']

        else:
            all_texts =  "\n".join(all_texts)
            print("Texts")
            print(all_texts)
            return ollama.generate('mistral', prompt=f"Given the question: {question}, find the answer from the following text. \n ###Text: {all_texts}")['response']
        
    