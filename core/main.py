from core import SemanticSimilarity, KeywordExtractor, KG, TripletGenerator

folder_path = "/home/niamatzawad/niamatzawad/Datasets/UTDBox/NYT Files/1-1000"
# folder_path = "/home/niamatzawad/niamatzawad/Datasets/UTDBox/demoNY/splitted"

s = SemanticSimilarity()
kwe = KeywordExtractor()
kg = KG()

folder_objects = os.listdir(folder_path)
for i, filename in enumerate(folder_objects):
    if os.path.isfile(os.path.join(folder_path, filename)):
        with open(os.path.join(folder_path, filename), 'r') as file:
            text = file.read()
            text_splitted = [text[i * 1900:(i+1) *1900] for i in range(len(text)//1900)]
            start_text_time = time.time()
            for text_s in tqdm(text_splitted):
                start_subtext_time = time.time()
                try:
                    named_entities = kwe.extract_named_entities(text_s)
                except Exception as e:
                    continue
                s.generate_reference(named_entities)
                t = TripletGenerator(text_s, s, kwe, os.path.join(folder_path, filename))
                triplets = t.get_triplets()
                kg.add_triplets(triplets)
                print(triplets)
                print(f"Time to create triplets for current subtext - {time.time() - start_subtext_time}")
                
        print(f"Finished processing {i+1}/ {len(folder_objects)} files")
        print(f"Total time taken for current text - {time.time() - start_text_time}")
        print("---")
            
kg.triplets