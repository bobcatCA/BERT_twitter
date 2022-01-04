#!/usr/bin/env python
# Twitter Auth.py

import json
from twitter_query import *
from twitter_dataCrunch import *

def main():
    # Inputs for the request
    keyword = "xbox"
    start_time = "2021-12-31T00:00:00.000Z"
    end_time = "2022-01-02T00:00:00.000Z"
    max_results = 10
    query_params = {'query': keyword,
                    'start_time': start_time,
                    'end_time': end_time,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    url = create_url()
    json_response = connect_to_endpoint(url, query_params)
    print(json.dumps(json_response, indent=4, sort_keys=True))
    append_to_csv(json_response, "v2query_tweets.csv")


if __name__ == "__main__":
    main()
