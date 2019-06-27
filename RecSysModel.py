class RecSysModel:

    def __init__(self, results):
        self.results = results

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        return self.results
