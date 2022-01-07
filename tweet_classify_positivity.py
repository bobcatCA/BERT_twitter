# !/usr/bin/env python
# coding: utf-8
# tweet_classify_positivity.py

import pandas as pd
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from tweet_classification_functions import *

# Load tweets into Pandas DataFrame
df = pd.read_csv('kaggle_1.6MMtwtr_noemoticon.csv', encoding='latin1',
                        names=['category', 'id', 'date', 'query', 'user', 'text'])
# df = pd.read_csv('12.30_tweets.csv',
#                  names=['author_id', 'created_at', 'geo', 'tweet_id', 'lang', 'like_count', 'quote_count',
#                       'reply_count', 'retweet_count', 'source', 'text'])
df = df.sample(n=int(100), random_state=1)  # Downsample to reduce time

# Get all possible labels - this might be pedantic for only binary classification
label_dict = {}
if hasattr(df, 'category'):
    labeled_dataset = True
    possible_labels = df.category.unique()  # labels are the unique values in category
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
        df['label'] = df.category.replace(label_dict)
        pass
else:
    # Easiest was to hard-code the labels we want; could think of a more elegant way in future
    labeled_dataset = False
    label_dict['positive'] = 0  # Positive label int=0 for logits
    label_dict['negative'] = 1  # Positive label int=1 for logits
    df['label'] = 0  # Set to zero then exclude accuracy calculations - find better way in future
    pass


# Loading Tokenizer and Encoding our Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encoded_data = tokenizer.batch_encode_plus(
    df.text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    truncation=True,
    padding='max_length',
    # pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

# Assemble ids, attention mask, and labels into Tensor Dataset
input_ids = encoded_data['input_ids']  #
attention_masks = encoded_data['attention_mask']  # Contains the text vs. blank end of the tweet

labels = torch.tensor(df.label.values)
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, sampler=None)  # Pedantic for not sampling but keep for now

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

# Load pre-trained model
model.load_state_dict(torch.load('BERT_ft_epoch2_posneg10k.model', map_location=torch.device('cpu')),
                      strict=False)

# Get results
if labeled_dataset:
    _, predictions, true_vals = evaluate_single(model, dataloader)
    accuracy_per_class(predictions, true_vals, label_dict)
else:
    _, predictions, _ = evaluate_single(model, dataloader)
    pass

# Insert predictions (logits) into original df and save as csv.
df['label0_logit'] = predictions[:, 0]
df['label1_logit'] = predictions[:, 1]
df.to_csv('tweets_with_logits.csv')




