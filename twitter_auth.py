#!/usr/bin/env python
# Twitter Auth.py

import json
from twitter_query import *
from twitter_dataCrunch import *

def main():
    # Inputs for the request
    first_connect = True
    keyword = '(corona OR covid OR c19) lang:en'
    start_time = "2021-12-30T00:10:00.000Z"
    end_time = "2021-12-30T00:18:00.000Z"
    max_results = 100
    next_token = {}
    query_params = {
                    'query': keyword,
                    'start_time': start_time,
                    'end_time': end_time,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': next_token}
    url = create_url()
    num_tweets = 0

    while True:
        # Optional section, to limit usage of tweet allowance unintentionally
        if num_tweets > 10000:
            print('unintended runaway, exited program')
            break

        json_response = connect_to_endpoint(url, query_params)  # Get JSON objects basend on query and URL
        # print(json.dumps(json_response, indent=4, sort_keys=True))
        append_to_csv(json_response, "12.30_tweets.csv")
        num_tweets = num_tweets + len(json_response['data'])
        if 'next_token' in json_response['meta']:  # Check if there is a next_token key in the meta dictionary
            next_token = json_response['meta']['next_token']
            query_params['next_token'] = next_token
        else:
            next_token = {}
            print('no more tweets to retrieve, exiting')  # If not, this is the last page
            pass
        # Check the length of the token, this is a bit overkill since in theory a returned value for token_length
        # should always be >0 in length
        try:
            next_token_length = len(next_token)
            if next_token_length == 0:
                break
            else:
                pass
        except NameError:
            print('NameError: variable "next_token" is not defined')
            break
        pass

    print('Total tweets collected: ', num_tweets)


if __name__ == "__main__":
    main()
