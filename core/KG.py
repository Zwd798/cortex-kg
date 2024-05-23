from typing import List
class KG:
    def __init__(self):
        self.triplets = []

    def add_triplets(self, triplets : List):
        self.triplets.extend(triplets)
