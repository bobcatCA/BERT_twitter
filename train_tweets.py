# !/usr/bin/env python
# coding: utf-8
# train_tweets.py
# Load labeled dataset and use to finetune BERT-pretrained model for sentiment analysis

import pandas as pd
import random
from tqdm import tqdm
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
from tweet_classification_functions import *


# Load tweets into df
df = pd.read_csv('kaggle_1.6MMtwtr_noemoticon.csv', encoding='latin1',
                        names=['category', 'id', 'date', 'query', 'user', 'text'])
df = df.sample(n=int(1E4), random_state=1)
df.category = df.category.replace([0, 4], ['positive', 'negative'])

# Transform df into useful objects to feed into model (see data_initializer() function)
tokenizer, df, encoded_data_train, encoded_data_val, label_dict = data_initialization(df)

# Assemble ids, attention mask, and labels into Tensor Dataset
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

# Turn them into Tensor Datasets
dataset_train = TensorDataset(input_ids_train,
                              attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val,
                              attention_masks_val, labels_val)

# Setting up BERT Pretrained Model
# Load pre-trained model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False)

# ## Task 6: Creating Data Loaders for training and val sets
batch_size = 4 #32
dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size)

dataloader_val = DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size=32)

# Setting Up Optimizer and Scheduler
epochs = 10  # 10 seems to over-fit
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)  # Adam W optimizer for dynamic learning rate
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

# Seeding is optional, helps generate repeatable results
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Device configuration to CPU/GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Enter main training loop. Go over range of epochs
for epoch in tqdm(range(1, epochs+1)):
    model.train()  # Put model into train mode (vs
    loss_train_total = 0  # Initialize training loss

    # Set TQDM progress bar, set to refresh at same position each iteration
    progress_bar = tqdm(dataloader_train,
                        desc='Epoch {:1d}'.format(epoch),
                        position=0,
                        leave=True,
                        disable=False)

    for batch in progress_bar:
        model.zero_grad()  # important to reset the Bias/Weight gradients between passes!
        batch = tuple(b.to(device) for b in batch)  # Store the input id's, attention masks, and labels
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }

        outputs = model(**inputs)  # Put the inputs through the model
        loss = outputs[0]  # Get loss for this round
        loss_train_total += loss.item()
        loss.backward()  # put loss backwards prop

        # Clip the gradients to prevent explosion!
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Take steps and set progress bar
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'training loss': '{:.3f}'.format(loss.item()/len(batch))})
    # Save model at the end of each epoch
    torch.save(model.state_dict(), f'BERT_ft_epoch{epoch}_binary.model')

    # Display epoch/loss information
    tqdm.write('\nEpoch {epoch}')
    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    val_loss, predictions, true_vals = evaluate(model, dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (weighted): {val_f1}')


print('done')


