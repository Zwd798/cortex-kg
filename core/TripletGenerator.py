import ollama

class TripletGenerator:
    def __init__(self, text : str, s : SemanticSimilarity, e : KeywordExtractor, doc_name : str):
        self.s = s
        self.e = e
        self.doc_name = doc_name
        self.text = text
        self.pattern = r"\((.*?)\)"
        self.model_name = "mistral"
        self.template = f"""Task:Generate triplets from the following text.
            Instructions'
            The triplet should be in the format (<subject>, <relation type>, <object>)
            The subject and object should refer to specific entities.
            The relation type should refer to an action
            
            The text is:
            {self.text}"""
        
        self.triplets = self._extract_triplets()

    def _get_entities_in_phrase(self, phrase) -> List:
        results = self.e.extract_named_entities(phrase)
        return results if results else [phrase]

        
    def _link_entities(self,subj,rel,obj):
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
                # s = re.sub(r'\s', '_', s.strip())
                # rel = re.sub(r'\s', '_', rel.strip())
                # o = re.sub(r'\s', '_', o.strip())
                s,rel,o = self.refine(s),self.refine(rel),self.refine(o)
                results.append((s,rel,o))
        return results

    def _add_doc_name_to_triplets(self, triplets):
        
        return [(w,x,y,z) for (w,x,y),z in zip(triplets,([self.doc_name] * len(triplets)))]

    def _add_doc_node(self, triplets):
        doc_triplets = []
        doc_name = re.sub(r'[\/]', '_', self.doc_name)
        doc_name = self.refine(doc_name)
        for t in triplets:
            doc_triplets.append((t[0],"filepath",doc_name))
        triplets.extend(doc_triplets)
        return triplets
        
    def _extract_triplets(self):
        triplet_results = []
        
        response = ollama.generate(model=self.model_name, prompt=self.template)
        
        for line in response['response'].split("\n"):
            matches = re.findall(self.pattern, line)
            if not matches:
                continue
            triplet = matches[0].split(",")
            if len(triplet) < 3:
                continue
            subj,rel,obj = triplet[0]," ".join(triplet[1:-1]), triplet[-1]
            subj,rel,obj = self._link_entities(subj, rel, obj)
            refined_triplets = self._refine_triplets(subj,rel,obj)
            # refined_triplets = self._add_doc_name_to_triplets(refined_triplets)
            refined_triplets = self._add_doc_node(refined_triplets)
            triplet_results.extend(refined_triplets)
            # triplet_results.append((subj.strip(),rel.strip(),obj.strip()))
        
        return triplet_results
    

        
    def get_triplets(self):
        return self.triplets