# tweet_classification_functions.py
# Contains a bunch of functions to be used for BERT learning/classification

# Libraries to be imported
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from transformers import BertTokenizer


def data_initialization(df):
    # Get all possible labels - this might be pedantic for only binary classification
    possible_labels = df.category.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df['label'] = df.category.replace(label_dict)

    # Split dataset into training/validation
    x_train, x_val, y_train, y_val = train_test_split(df.index.values,
                                                      df.label.values,
                                                      test_size=0.15,
                                                      random_state=17,
                                                      stratify=df.label.values)

    # Make a column that tells us what set each row has been assigned to (train or val)
    df['data_type'] = ['not_set'] * df.shape[0]
    df.loc[x_train, 'data_type'] = 'train'
    df.loc[x_val, 'data_type'] = 'val'
    df.groupby(['category', 'label', 'data_type']).count()

    # Load Tokenizer and Encode our Data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type == 'train'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        # pad_to_max_length=True,
        max_length=256,
        return_tensors='pt')

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type == 'val'].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        # pad_to_max_length=True,
        max_length=256,
        return_tensors='pt')
    return tokenizer, df, encoded_data_train, encoded_data_val, label_dict

def f1_score_func(preds, labels):
    """
    Compute F1 score given arrays of labels and predictions
    :param preds: numpy array
    :param labels: numpy array
    :return: labels, predictions
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels, label_dict):
    """
    Compute accuracy, given predictions, labels
    :param preds: numpy array
    :param labels: numpy array
    :param label_dict: dictionary
    :return: None, just prints results
    """
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')
        pass
    return


def evaluate(model, dataloader):
    """
    Evaluates a tensor dataset of tweets (dataloader), using a BERT model (model). Built to 
    take in and process an entire data set at once.
    
    :param model: fully trained BERT model object
    :param dataloader: DataLoader instance (see PyTorch DataLoader documentation)
    :return: average loss (float), predictions (numpy array), true_vals, or labels, (numpy array)
    """
    
    model.eval()  # Set model to evaluate mode (vs. train mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CPU for this computer
    model.to(device)
    loss_val_total = 0  # Initialize loss, predictions
    predictions, true_vals = [], []

    for batch in tqdm.tqdm(dataloader):  # Loop over all batches in dataloader

        # Make batch a tuple of input id's, attention mask, and label tensors
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        # no_grad() to disable gradient calculation - this will save computational resources since
        # we are not backpropagating here, just forward
        with torch.no_grad():
            outputs = model(**inputs)

        # Get results
        loss = outputs[0]  # Loss
        logits = outputs[1]  # Classification
        loss_val_total += loss.item()  # Loss total
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def evaluate_single(model, dataloader):
    """
    Evaluates a tensor dataset of tweets (dataloader), using a BERT model (model). Built to take batches

    :param model: fully trained BERT model object
    :param dataloader: DataLoader instance (see PyTorch DataLoader documentation)
    :return: average loss (float), predictions (numpy array), true_vals, or labels, (numpy array)
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_val_total = 0
    predictions, true_vals = [], []

    inputs = {'input_ids':      dataloader.dataset.tensors[0],
              'attention_mask': dataloader.dataset.tensors[1],
              'labels':         dataloader.dataset.tensors[2],
             }

    # no_grad() to disable gradient calculation - this will save computational resources since
    # we are not backpropagating here, just forward
    with torch.no_grad():
        outputs = model(**inputs)

        # Get results
    loss = outputs['loss']  # Loss
    logits = outputs['logits']  # Classification
    loss_val_total += loss.item()  # Loss total
    logits = logits.detach().cpu().numpy()
    label_ids = inputs['labels'].cpu().numpy()
    predictions.append(logits)
    true_vals.append(label_ids)
    loss_val_avg = loss_val_total/len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


