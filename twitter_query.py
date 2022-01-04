# twitter_query.py

from twitter_pulltocsv import *
import requests
import os


def create_url():
    """
    :return: url in the API in which to search
    """
    url = 'https://api.twitter.com/2/tweets/search/recent'
    # Update this function in the future if necessary (may or may not need to)
    return url


def bearer_oauth(r):
    """
    :param r:  PreparedRequest (request library).
    :return:  PreparedRequest object with bearer token authentication and API name (to be used with URL
    """
    # Note that the bearer token is used in authentication. In order to not reveal the bearer token, it is saved to
    # local environment variables
    bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
    r.headers["Authorization"] = f"Bearer {bearer_token}"  # Bearer token
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def connect_to_endpoint(url, params):
    """
    :param url: URL to search API. Generated in create_url() functon
    :param params: URL Query, containing desired operators
        EXAMPLE:
        params = {'query': keyword,
            'start_time': start_time,
            'end_time': end_time,
            'max_results': max_results,
            'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
            'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at
            'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
            'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
            'next_token': {}}
        See Twitter Developer page for more documentation on building queries:
        https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
    :return: JSON tweet object (if successful)
    """
    # Get response based on URL, parameters (query), and authentication
    response = requests.request("GET", url, params=params, auth=bearer_oauth)
    print(response.status_code)  # 200 if successful
    if response.status_code != 200:  # Raise exception and error message if response is not successful
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()



