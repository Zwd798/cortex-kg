import ollama
from SemanticSimilarity import SemanticSimilarity
from KeywordExtractor import KeywordExtractor
from typing import List, Tuple, Union
import re

class TripletGenerator:

    def __init__(self, text : str, s : SemanticSimilarity, e : KeywordExtractor, doc_name : str, named_entities, template=None, ):
        self.s = s
        self.e = e
        self.doc_name = doc_name
        self.text = text
        self.pattern = r"\((.*?)\)"
        self.pattern_text = r"Text:\s(.*)"
        self.model_name = "mistral"
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

    
    def _extract_triplets(self):
        triplet_results = []
        self.template = f"""Task:Given the entities {self.named_entities}, loop across each of the entities. If triplets can be generated on the urrent entity, generate them. If the entitiy does not exist in the text, reply with 'None'. For each triplet, mention the text from which the triplet was extracted
            ###Instructions:
            The triplet should be in the format (<subject>, <relation type>, <object>), where the subject must be one of the entities in {self.named_entities}.
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
    
        print('the named entities are ')
        print(self.named_entities)
        
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
    

        
    def get_triplets(self):
        return self.triplets
        