import pandas as pd

class RecSysEvaluator:
    def __init__(self, test_interactions, train_interactions):
        self.train_interactions = train_interactions
        self.test_interactions = test_interactions

    def get_items_interacted(self, person_id, interactions):
        # Get the user's data and merge in the movie information.
        interacted_items = interactions.loc[person_id]['contentId']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

    def recall_at_k(self, person_interactions_items, recommendations, k=10):
        n_rel = len(person_interactions_items)
        n_rel_and_rec_k = sum((content_id in person_interactions_items)
                              for content_id in recommendations['contentId'][:k])
        return n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        
    def evaluate_model_for_user(self, model, person_id):
        # Getting a ranked recommendation list from a model for a given user
        items_to_ignore = self.get_items_interacted(person_id, self.train_interactions)
        recommendations = model.recommend_items(person_id, items_to_ignore, topn=10000000000)
        person_interactions = self.test_interactions.loc[person_id]
        if type(person_interactions['contentId']) == pd.Series:
            person_interactions_items = set(person_interactions['contentId'])
        else:
            person_interactions_items = set([int(person_interactions['contentId'])])

        recall_at_5 = self.recall_at_k(person_interactions_items, recommendations, 5)
        recall_at_10 = self.recall_at_k(person_interactions_items, recommendations, 10)
        recall_at_25 = self.recall_at_k(person_interactions_items, recommendations, 25)

        person_metrics = {
            'interacted_count': len(person_interactions_items),
            'recall@5': recall_at_5,
            'recall@10': recall_at_10,
            'recall@25': recall_at_25
        }
        return person_metrics

    def evaluate_model(self, model):
        # print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(self.test_interactions.index.unique().values)):
            # if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        results = pd.DataFrame(people_metrics) \
            .sort_values('interacted_count', ascending=False)

        global_recall_at_5 = results['recall@5'].sum() / float(len(results))
        global_recall_at_10 = results['recall@10'].sum() / float(len(results))
        global_recall_at_25 = results['recall@25'].sum() / float(len(results))

        metrics = {
            'modelName': 'Test',
            'recall@5': global_recall_at_5,
            'recall@10': global_recall_at_10,
            'recall@25': global_recall_at_25
        }

        return metrics, results
