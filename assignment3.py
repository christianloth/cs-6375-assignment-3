import pandas as pd
import re
from kmeans_tweets import KMeansTweets
from kmeans_numeric import KMeansNumeric

URL = 'https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/Health-Tweets/usnewshealth.txt'
K_VALS_FOR_OUTPUT = [2, 3, 4, 5, 10]  # Modify this list to include the k values you want to output


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
    #preprocessed_tweets = preprocess_tweets(tweets_df.head(5))

    table_rows = []

    for k in K_VALS_FOR_OUTPUT:
        kmeans = KMeansTweets(K=k)
        kmeans.fit(preprocessed_tweets)
        cluster_assignments = kmeans.find_closest_centroids(preprocessed_tweets)

        # Calculate SSE
        sse = kmeans.calculate_sse()

        # Calculate the size of each cluster
        cluster_sizes = kmeans.calculate_cluster_sizes()

        # Store the results in a tuple
        table_rows.append((k, sse, cluster_sizes))

    # Now print the table
    print(f"{'Value of k':<10} | {'SSE':<10} | {'Size of each cluster':<30}")
    print('--------------------------------------------------------------------------------------')

    for k, sse, sizes in table_rows:
        sizes_str = ", ".join(f"{i}: {size} tweets" for i, size in sizes.items())
        print(f"{k:<10} | {sse:<10.3f} | {sizes_str:<30}")
