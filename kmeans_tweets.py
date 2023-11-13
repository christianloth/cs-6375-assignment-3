from typing import Dict, List, Optional, Set

import numpy as np
import random


def jaccard_distance(setA: Set[str], setB: Set[str]) -> float:
    """
    :param setA: set A
    :param setB: set B
    :return: int the Jaccard distance between two sets
    """
    intersection = len(setA.intersection(setB))
    union = len(setA.union(setB))
    return 1 - intersection / union


def tweet_to_set(tweet: str) -> Set[str]:
    """

    :param tweet: string of the tweet
    :return: set of words in tweet
    """
    return set(tweet.lower().split())


def find_centroid(cluster: List[Set[str]]) -> Optional[Set[str]]:
    """
    Finds the centroid for a cluster by calculating the tweet that has the
    smallest sum of distances to all other tweets in the cluster.

    :param cluster: list of sets of words in each tweet
    :return: set of words in the centroid tweet
    """

    if not cluster:
        return None

    # Cache distances in a HashMap/dictionary to reduce overhead from O(n*(n-1)) to O(n*(n-1)/2))
    distance_cache = {}
    for i, tweet in enumerate(cluster):
        for j, tweet_to_compare in enumerate(cluster):
            if j > i:  # Only compute each pair once
                distance_cache[frozenset([i, j])] = jaccard_distance(tweet, tweet_to_compare)

    # Calculate the sum of distances from each tweet to others in the cluster
    min_distance_sum = float('inf')
    centroid = None
    for i, tweet in enumerate(cluster):
        distance_sum = sum(distance_cache[frozenset([i, j])] for j in range(len(cluster)) if i != j)
        if distance_sum < min_distance_sum:
            min_distance_sum = distance_sum
            centroid = tweet

    return centroid


class KMeansTweets:
    def __init__(self, K, max_iters=100, random_seed=3) -> None:
        self.K = K  # number of clusters to group the data into
        self.max_iterations = max_iters
        self.clusters = [[] for _ in range(K)]
        self.centroids = []  # There will be one centroid for each cluster/K
        self.random_seed = random_seed

    def perform_clustering(self, tweets: List[str]) -> None:
        """
        Perform K-means clustering on the given tweets.
        :param tweets: set of tweet strings
        """
        # Convert tweets to sets of words
        tweets = [tweet_to_set(tweet) for tweet in tweets]

        random.seed(self.random_seed)
        self.centroids = random.sample(tweets, self.K)  # start out guessing centroids

        for i in range(self.max_iterations):
            # First, we will assign tweets to the closest centroids
            self.clusters = [[] for _ in range(self.K)]
            for tweet in tweets:
                distances_from_centroids = [jaccard_distance(tweet, centroid) for centroid in self.centroids]  # calculate Jaccard distance for each tweet from each centroid
                closest_centroid_idx = np.argmin(distances_from_centroids)
                self.clusters[closest_centroid_idx].append(tweet)

            # Update centroids
            new_centroids = []
            for cluster in self.clusters:
                if cluster:  # check if the cluster isn't empty
                    new_centroid = find_centroid(cluster)
                    new_centroids.append(new_centroid)
                else:  # if empty, randomly reinitialize its centroid
                    new_centroids.append(random.choice(tweets))

            # Check for convergence. If centroids didn't change in the round, then we're done
            if set(map(tuple, new_centroids)) != set(map(tuple, self.centroids)):  # if round centroids changed
                self.centroids = new_centroids
            else:  # if round centroids stayed the same
                break

    def find_closest_centroids(self, tweets: List[str]) -> List[int]:
        """
        Assigns each tweet to its closest centroid.
        :param tweets: set of tweet strings
        :return: list of centroid indexes representing the closest centroid for each tweet.
        """
        # Convert tweets to sets of words
        tweets = [tweet_to_set(tweet) for tweet in tweets]

        centroid_predictions = []
        for tweet in tweets:
            distances_from_centroids = [jaccard_distance(tweet, centroid) for centroid in self.centroids]  # calculate Jaccard distance for each tweet from each centroid
            closest_centroid_idx = np.argmin(distances_from_centroids)  # index of the closest centroid, and the smallest value in distances_from_centroids
            centroid_predictions.append(closest_centroid_idx)  # add the index to the list of predictions
        return centroid_predictions

    def calculate_sse(self) -> float:
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
                sse += jaccard_distance(tweet, centroid) ** 2

        return sse

    def calculate_cluster_sizes(self) -> Dict[int, int]:
        """
        Calculate the size (number of tweets) of each cluster.
        :return: dictionary where the key is the cluster index and the value is the size.
        """
        cluster_sizes = {i: len(cluster) for i, cluster in enumerate(self.clusters)}
        return cluster_sizes
