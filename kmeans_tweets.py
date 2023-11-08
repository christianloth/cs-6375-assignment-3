import numpy as np
import random

class KMeansTweets():
    def __init__(self, cluster_size, max_iters=100, random_seed=3):
        self.cluster_size = cluster_size
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(cluster_size)]
        self.centroids = []
        self.sse = None
        self.random_seed = random_seed

    def jaccard_distance(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1 - intersection / union

    def tweet_to_set(self, tweet):
        return set(tweet.lower().split())

    def find_centroid(self, cluster):
        min_distance = float('inf')
        centroid = None
        for tweet in cluster:
            distance_sum = sum(self.jaccard_distance(tweet, other_tweet)
                               for other_tweet in cluster if other_tweet != tweet)
            if distance_sum < min_distance:
                min_distance = distance_sum
                centroid = tweet
        return centroid

    def fit(self, tweets):
        # Convert tweets to sets of words
        tweets = list(map(self.tweet_to_set, tweets))

        # Init centroids
        random.seed(self.random_seed)
        self.centroids = random.sample(tweets, self.cluster_size)

        for _ in range(self.max_iters):
            # Assign tweets to the closest centroids
            self.clusters = [[] for _ in range(self.cluster_size)]
            for tweet in tweets:
                distances = [self.jaccard_distance(tweet, centroid) for centroid in self.centroids]
                closest_centroid_index = np.argmin(distances)
                self.clusters[closest_centroid_index].append(tweet)

            # Update centroids
            new_centroids = []
            for cluster in self.clusters:
                if cluster:  # avoid division by zero
                    new_centroid = self.find_centroid(cluster)
                    new_centroids.append(new_centroid)
                else:  # if a cluster is empty, randomly reinitialize its centroid
                    new_centroids.append(random.choice(tweets))

            # Check for convergence (if centroids don't change)
            if set(map(tuple, new_centroids)) == set(map(tuple, self.centroids)):
                break
            else:
                self.centroids = new_centroids

    def predict(self, tweets):
        tweets = list(map(self.tweet_to_set, tweets))
        predictions = []
        for tweet in tweets:
            distances = [self.jaccard_distance(tweet, centroid) for centroid in self.centroids]
            closest_centroid_index = np.argmin(distances)
            predictions.append(closest_centroid_index)
        return predictions

    def calculate_sse(self):
        """
        Formula parameters:

        Ci: set of points in cluster i
        mi: centroid of cluster i
        x: tweet point
        SSE = sum of the squared distance of each point in cluster i from the centroid mi
        """

        sse = 0
        for i, cluster in enumerate(self.clusters):
            centroid = self.centroids[i]
            for tweet in cluster:
                sse += self.jaccard_distance(tweet, centroid) ** 2

        return sse



from sklearn.cluster import KMeans

class KMeansNumeric():
    def __init__(self, cluster_size, max_iters=100, random_seed=3):
        self.cluster_size = cluster_size
        self.max_iters = max_iters
        self.model = KMeans(n_clusters=cluster_size,
                            max_iter=max_iters,
                            random_state=random_seed)

    def fit(self, data):
        # Assuming 'data' is a 2D array or a DataFrame with numerical values
        self.model.fit(data)

    def predict(self, data):
        # Returns the cluster index for each sample
        return self.model.predict(data)

    def calculate_sse(self):
        # Returns the SSE (inertia) of the current model
        return self.model.inertia_
