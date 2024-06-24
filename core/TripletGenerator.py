import ollama
from SemanticSimilarity import SemanticSimilarity
from KeywordExtractor import KeywordExtractor
from typing import List, Tuple, Union
import re
from openie import StanfordOpenIE

class TripletGenerator:

    def __init__(self, text : str, s : SemanticSimilarity, e : KeywordExtractor, doc_name : str, named_entities, template=None, ):
        self.s = s
        self.e = e
        self.doc_name = doc_name
        self.text = text
        self.pattern = r"\((.*?)\)"
        self.pattern_text = r"Text:\s(.*)"
        self.model_name = "llama3"
        self.properties = {
    'openie.affinity_probability_cap': 2 / 3,
}
        self.template = f"""Task:Generate triplets from the following text. For each triplet, mention the text from which the triplet was extracted
                ###Instructions:
                The triplet should be in the format (<subject>, <relation type>, <object>)
                The subject and object should refer to specific entities.
                The relation type should refer to an action.
                The text should be the paragraph from where the triplet was extracted. Just extract the sentence verbatim and do not make modifications
            

             ###Example answer:
             (A, likes, B)
             Text: In the year 1993 A started liking B
             
            ###The text is:
            {self.text}"""
        self.named_entities = named_entities
        self.triplets = self._extract_triplets()
        

    def _get_entities_in_phrase(self, phrase) -> List:
        results = self.e.extract_named_entities(phrase)
        return results if results else [phrase]

        
    def _link_entities(self,subj,rel,obj): #This needs to be worked on as of now I am just using a simple cosine similarity search
        available_subj = self.s.get_most_similar_word(subj)
        if available_subj:
            subj = available_subj

        available_rel = self.s.get_most_similar_relation(rel)
        if available_rel:
            rel = available_rel
        else:
            self.s.add_relation(rel)
            
        available_obj = self.s.get_most_similar_word(obj)
        if available_obj:
            obj = available_obj

        return subj, rel, obj

    def refine(self, phrase):
        phrase = phrase.strip()
        phrase = phrase.lower()
        phrase = re.sub(r'[,\.!?]', '', phrase)
        phrase = re.sub(r'[^a-z0-9\s]', ' ', phrase)
        phrase = re.sub(r"^\s*(?:the|a)\s+", "", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r'\s+', '_', phrase)
        return phrase
        
    def _refine_triplets(self, subj, rel, obj): #for example for a triplet Bill and John - played - Baseball, it will be split into two triplets with subjects being Bill and John 
        subj_nouns = self._get_entities_in_phrase(subj)
        obj_nouns = self._get_entities_in_phrase(obj)
        results = []
        for s in subj_nouns:
            for o in obj_nouns:
                # s,rel,o = self.refine(s),self.refine(rel),self.refine(o)
                results.append((s,rel,o))
        return results

    def _add_doc_name_to_triplets(self, triplets):
        
        return [(w,x,y,z) for (w,x,y),z in zip(triplets,([self.doc_name] * len(triplets)))]

    def _get_summary(self):
        prompt = f"""Task:Given the text {self.text}, generate a summary"""
        summary = ollama.generate(model="mistral", prompt=prompt)["response"]
        print("The summary is")
        print(summary)
        return summary

    
    def _add_label_nodes(self, triplets, summary):
        prompt = f"""Task:Given the text {self.text}, generate some labels which best classify the theme of the text"""
        labels = ollama.generate(model="mistral", prompt=prompt)["response"]
        pattern = r'\d+\.\s*(.+)'
        matches = re.findall(pattern, labels)
        print("The matches are")
        print(matches)
        doc_triplets = []
        for m in matches:
            doc_triplets.append((m, "category", self.doc_name))
            doc_triplets.append((m, "category", summary))
            doc_triplets.append((self.doc_name, "category", m)) #bidirectional
            doc_triplets.append((summary, "category", m))
        
        triplets.extend(doc_triplets)
        return triplets


    def _add_doc_node(self, triplets):
        doc_triplets = []
        # doc_name = re.sub(r'[\/]', '_', self.doc_name)
        # doc_name = self.refine(doc_name)
        for t in triplets:
            # doc_triplets.append((t[0],"filepath",doc_name))
            doc_triplets.append((t[0],"filepath",self.doc_name))
        triplets.extend(doc_triplets)
        return triplets

    
    def decompose_triplets(self, triplets): #if a triplet is of the form (subject, rel1, rel2 , .. , obj, doc_node), decompose it to (subject, rel1, obj, doc_node), (subject, rel2, obj, doc_node)
        for i, t in enumerate(triplets):  #(subject, rel, rel , .. , obj, doc_node)
            if len(t[1:-2]) > 1:
                temp_t = t
                triplets = triplets[:i] + triplets[i+1:]
                for rel in temp_t[1:-2]:
                    triplets.append((temp_t[0], rel, temp_t[-2],temp_t[-1]))
        return triplets

    
    def _extract_triplets_based_on_single_entity(self):
        triplet_results = []
        for en in self.named_entities:
            self.template = f"""Task:Given the entity {en}, generate triplets for it from the text. If the entity does not exist in the text, reply with 'None'. For each triplet, mention the text from which the triplet was extracted
                ###Instructions:
                The triplet should be in the format (<subject>, <relation type>, <object>), where the subject must be the entity {en}.
                The object should refer to a specific entity.
                The relation type should refer to an action.
                The text should be the paragraph from where the triplet was extracted. Just extract the sentence verbatim and do not make modifications
            
    
             ###Example answer:
             Entity: A
             Text: A started liking B
             (A, likes, B)
    
             Entity: X
             Text: A started liking B
             None
             
             
            ###The text is:
            {self.text}"""
            print("the text is ")
            print(self.text)
        
            print('the named entity is ')
            print(en)
            
            response = ollama.generate(model=self.model_name, prompt=self.template)
            
            for line in response['response'].split("\n"):
                if 'None' in line:
                    print('none found in line')
                    continue
                matches = re.findall(self.pattern, line)
                matches_text = re.findall(self.pattern_text, line)
                if not matches:
                    if matches_text:
                        if len(triplet_results) > 1:
                            triplet_results[-2] = triplet_results[-2] + (matches_text[0],)  #Attach text to triplet
                    continue
                triplet = matches[0].split(",")
                if len(triplet) < 3:
                    continue
    
                
                subj,rel,obj = triplet[0]," ".join(triplet[1:-1]), triplet[-1]
                subj,rel,obj = self._link_entities(subj, rel, obj)
                refined_triplets = [(subj, rel, obj)]
                # refined_triplets = self._add_doc_name_to_triplets(refined_triplets)
                refined_triplets = self._add_doc_node(refined_triplets)
                refined_triplets = self.decompose_triplets(refined_triplets) #This should always be called after adding the doc node as it assumes 4 entities per tuple
                triplet_results.extend(refined_triplets)
                # triplet_results.append((subj.strip(),rel.strip(),obj.strip()))
    
        print("the triplets are")
        print(triplet_results)
        return triplet_results
        

    def _triplet_generation(self, response):
        triplet_results = []
        for line in response['response'].split("\n"):
            if 'None' in line:
                print('none found in line')
                continue
            matches = re.findall(self.pattern, line)
            matches_text = re.findall(self.pattern_text, line)
            if not matches:
                if matches_text:
                    if len(triplet_results) > 1:
                        triplet_results[-2] = triplet_results[-2] + (matches_text[0],)  #Attach text to triplet
                continue
            triplet = matches[0].split(",")
            if len(triplet) < 3:
                continue

            
            subj,rel,obj = triplet[0]," ".join(triplet[1:-1]), triplet[-1]
            subj,rel,obj = self._link_entities(subj, rel, obj)
            refined_triplets = [(subj, rel, obj)]
            # refined_triplets = self._add_doc_name_to_triplets(refined_triplets)
            refined_triplets = self._add_doc_node(refined_triplets)
            refined_triplets = self.decompose_triplets(refined_triplets) #This should always be called after adding the doc node as it assumes 4 entities per tuple
            triplet_results.extend(refined_triplets)
            # triplet_results.append((subj.strip(),rel.strip(),obj.strip()))
        return triplet_results


    def _extract_triplets_second_pass(self, triplets):
        template = f"""Task:Given the entities {self.named_entities} , loop across each of the entities and generate triplets that are not in the list of triplets {triplets}. If triplets can be generated on the current entity, generate them. If the entity does not exist in the text, reply with 'None'. For each triplet, mention the text from which the triplet was extracted
            ###Instructions:
            The triplet should be in the format (<subject>, <relation type>, <object>), where the subject must be one of the entities in {self.named_entities}.
            The object should refer to a specific entity.
            The relation type should refer to an action.
            The text should be the paragraph from where the triplet was extracted. Just extract the sentence verbatim and do not make modifications

            ###The text is:
        {self.text}
            """
        response = ollama.generate(model=self.model_name, prompt=template)
        triplet_results = self._triplet_generation(response)
        print('triplet obtained after the first pass')
        print(triplets)
        print('here are the triplets obtained after the second pass')
        print(triplet_results)
        triplets += triplet_results
        return triplets
        
    def _extract_triplets(self):
        triplet_results = []
            
        with StanfordOpenIE(properties=self.properties) as client:
            # with open('/home/niamatzawad/niamatzawad/Datasets/HotpotQA/hotpot_1/row_0.txt', encoding='utf8') as r:
            #     corpus = r.read().replace('\n', ' ').replace('\r', '')
            triples_corpus = client.annotate(self.text)
            print('Corpus: %s [...].' % self.text[0:80])
            print('Found %s triples in the corpus.' % len(triples_corpus))
            for triple in triples_corpus:
                triplet_results.append((triple["subject"], triple["relation"], triple["object"]))
            print('[...]')
        
        print("the text is ")
        print(self.text)
    
        print('the named entities are ')
        print(self.named_entities)
                
        summary = self._get_summary()

        triplet_results.extend([(summary, "component", t[0]) for t in triplet_results]) #Add link between summary and triplets
        
        self._add_label_nodes(triplet_results,summary)
        print("the triplets are")
        print(triplet_results)
        
        return triplet_results
    
    # def disambiguate_triplets(self, triplets):
    #     visited_subjects = 
    #     for t in triplets:
    #         s,r,o = t
    #         if s in triplets:
                
            
        
    def get_triplets(self):
        return self.triplets
        