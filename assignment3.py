import pandas as pd
import re
from kmeans_tweets import KMeansTweets, KMeansNumeric

URL = 'https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/Health-Tweets/usnewshealth.txt'


def preprocess_tweets(df):
    def preprocess_tweet(tweet):
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = tweet.replace('#', '')
        tweet = tweet.lower()
        return tweet.strip()

    df = df.copy()
    df['tweet'] = df['tweet'].apply(preprocess_tweet)
    # Drop the columns we don't need
    df = df.drop(columns=['tweet_id', 'timestamp'])
    return df['tweet'].tolist()



#preprocessed_tweets = preprocess_tweets(tweets_df.head(5))


'''
kmeans = KMeansTweets(cluster_size=3)
kmeans.fit(preprocessed_tweets)

# Get cluster assignments for each tweet
cluster_assignments = kmeans.predict(preprocessed_tweets)

# Print out the cluster assignments for the first few tweets
for i, cluster in enumerate(cluster_assignments[:10]):
    print(f"Tweet {i}: Cluster {cluster}")

# Calculate and print the size of each cluster
cluster_sizes = {i: cluster_assignments.count(i) for i in range(kmeans.cluster_size)}
print("Cluster sizes:", cluster_sizes)

# Optionally, examine the centroids
# Convert sets back to strings to make them more readable
centroids = [' '.join(centroid) for centroid in kmeans.centroids]
for i, centroid in enumerate(centroids):
    print(f"Centroid {i}: {centroid}")

# Calculate SSE
print(kmeans.calculate_sse())

'''


if __name__ == '__main__':
    tweets_df = pd.read_csv(URL, sep='|', header=None, names=['tweet_id', 'timestamp', 'tweet'])  # read_csv so that it can download online file. Works with .txt here.
    preprocessed_tweets = preprocess_tweets(tweets_df)

    # Define a list for different values of k
    k_values = [2, 3, 4, 5, 10]  # Add any other k values you want to test

    # Initialize a list to store the results for each k
    results = []

    for k in k_values:
        # Run KMeans with the current value of k
        kmeans = KMeansTweets(cluster_size=k)
        kmeans.fit(preprocessed_tweets)
        cluster_assignments = kmeans.predict(preprocessed_tweets)

        # Calculate SSE
        sse = kmeans.calculate_sse()

        # Calculate the size of each cluster
        cluster_sizes = {i: cluster_assignments.count(i) for i in range(k)}

        # Store the results
        results.append((k, sse, cluster_sizes))

    # Now print the table
    print(f"{'Value of k':<10} | {'SSE':<10} | {'Size of each cluster':<30}")
    print('-' * 60)  # Print a separator line

    for k, sse, sizes in results:
        sizes_str = ", ".join(f"{i}: {size} tweets" for i, size in sizes.items())
        print(f"{k:<10} | {sse:<10.2f} | {sizes_str:<30}")






