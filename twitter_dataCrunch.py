# twitter_dataCrunch.py

import dateutil
from dateutil import parser
import csv


def append_to_csv(json_response, fileName):
    """
    :param json_response: JSON objects (tweets) from twitter v2 API
    :param fileName: string, of chosen filename (if existing or not)
    :return: None, just the csv will be created/modified
    """
    # Counter variable, just for display/debugging at the end
    counter = 0

    # Open csv file of fileName string, or create if it doesn't exist
    # Create writer object
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    # Loop through each tweet and retrieve the desired data
    for tweet in json_response['data']:
        author_id = tweet['author_id']  # 1. Author ID
        created_at = dateutil.parser.parse(tweet['created_at'])  # 2. Time created

        # 3. Geolocation - might not exist so check first
        if ('geo' in tweet):
            geo = tweet['geo']['place_id']
        else:
            geo = " "

        tweet_id = tweet['id']  # 4. Tweet ID
        lang = tweet['lang']  # 5. Language
        source = tweet['source']  # 6. Source
        text = tweet['text']  # 8. Tweet text - the actual status/text body of the tweet

        # 9. Tweet metrics
        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']

        # Assemble all data in a list
        tweet_data = [author_id, created_at, geo, tweet_id, lang, like_count, quote_count, reply_count, retweet_count, source,
               text]

        # Append the result to the CSV file
        csvWriter.writerow(tweet_data)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added: ", counter)
