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

    def jaccard_distance(self, setA, setB):
        """
        :param setA: hashset set A
        :param setB: hashset set B
        :return: int the Jaccard distance between two sets
        """
        intersection = len(setA.intersection(setB))
        union = len(setA.union(setB))
        return 1 - intersection / union

    def tweet_to_set(self, tweet):
        """

        :param tweet: string of the tweet
        :return: hashset of words in tweet
        """
        return set(tweet.lower().split())

    def find_centroid(self, cluster):
        """
        Finds the centroid for a specific cluster

        :param cluster: list of hashsets of words in each tweet
        :return: hashset of words in the centroid tweet
        """
        min_distance = float('inf')
        centroid = None
        for tweet in cluster:
            distance_sum = 0
            for tweet_to_compare in cluster:
                if tweet_to_compare != tweet:
                    distance_sum += self.jaccard_distance(tweet, tweet_to_compare)
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

        for i in range(self.max_iters):
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

    def calculate_cluster_sizes(self):
        """
        Calculate the size (number of tweets) of each cluster.
        Returns a dictionary where the key is the cluster index and the value is the size.
        """
        cluster_sizes = {i: len(cluster) for i, cluster in enumerate(self.clusters)}
        return cluster_sizes