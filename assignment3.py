import pandas as pd
import re

from matplotlib import pyplot as plt

from kmeans_tweets import KMeansTweets

# Constants
URL = 'https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/Health-Tweets/usnewshealth.txt'
K_VALS_FOR_OUTPUT = [2, 3, 4, 5, 10]  # User can modify this list to include the k values you want to output
RANDOM_SEED = 35  # User can modify this random seed too


def preprocess_tweets(df):
    def preprocess_tweet(tweet):
        tweet = re.sub(r'http\S+', '', tweet)  # remove the URL
        tweet = re.sub(r'@\w+', '', tweet)  # remove the @
        tweet = tweet.replace('#', '')  # remove the # in hashtags
        tweet = tweet.lower()  # convert to lowercase
        return tweet.strip()

    df = df.copy()
    df['tweet'] = df['tweet'].apply(preprocess_tweet)
    df = df.drop(columns=['tweet_id', 'timestamp'])  # remove the tweet_id and timestamp columns
    return df['tweet'].tolist()


if __name__ == '__main__':
    tweets_df = pd.read_csv(URL, sep='|', header=None, names=['tweet_id', 'timestamp', 'tweet'])  # read_csv so that it can download online file. Works with .txt here.
    preprocessed_tweets = preprocess_tweets(tweets_df)

    table_rows = []

    for k in K_VALS_FOR_OUTPUT:
        kmeans = KMeansTweets(K=k, random_seed=RANDOM_SEED)
        kmeans.perform_clustering(preprocessed_tweets)
        cluster_assignments = kmeans.find_closest_centroids(preprocessed_tweets)

        sse = kmeans.calculate_sse()
        cluster_sizes = kmeans.calculate_cluster_sizes()

        # Store the results in a tuple
        table_rows.append((k, sse, cluster_sizes))

    # Now print the table
    print(f"{'Value of k':<10} | {'SSE':<15} | {'Size of each cluster':<30}")
    print('--------------------------------------------------------------------------------------')

    for k, sse, cluster_sizes in table_rows:
        sizes_str = ", ".join(f"{i}: {num_tweets} tweets" for i, num_tweets in cluster_sizes.items())
        print(f"{k:<10} | {sse:<15.3f} | {sizes_str:<30}")

    # Plotting k vs SSE
    k_values = [row[0] for row in table_rows]
    sse_values = [row[1] for row in table_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sse_values, marker='o')
    plt.title('K vs SSE')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()