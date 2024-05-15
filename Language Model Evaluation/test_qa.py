#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
#from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import EarlyStoppingCallback, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import BertTokenizerFast, BertForSequenceClassification
import collections

def compute_f1(gold_toks, pred_toks):

  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return [int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)]
  if num_same == 0:
    return [0, 0, 0]
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return [precision, recall, f1]


import sys
sys.path.append(".")
sys.path.append("..")

model_name = "bert-base-uncased"
max_length = 512
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)


data = pd.read_csv('policy_train_data.csv', '\t')

gss = GroupShuffleSplit(train_size=.80, random_state = 2, n_splits=1)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


for train_idx, val_idx in gss.split(data.loc[:, data.columns != 'Label'], data['Label'], groups=data['QueryID']):
    train_ds = data.iloc[train_idx]
    val_ds = data.iloc[val_idx]


train_label = pd.factorize(train_ds.Label)[0] 
valid_label = pd.factorize(val_ds.Label)[0]

#train_label = 1 - train_label #because 0 should be relevant and 1 shoild be irrelevant
#valid_label = 1 - valid_label

count = 0

print("Encodings generation.")
train_encodings = tokenizer(train_ds['Query'].tolist(),train_ds['Segment'].tolist(), truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(val_ds['Query'].tolist(), val_ds['Segment'].tolist(), truncation=True, padding=True, max_length=max_length)
print("Encodings generated.")

train_dataset = Dataset(train_encodings, train_label) 
valid_dataset = Dataset(valid_encodings, valid_label)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, return_dict=True).to("cuda")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
output_dir='test_pqa_bert/pqa_bert_{}'.format(count),          
num_train_epochs=3,              
logging_dir='test_pqa_bert/logs_pqa_bert_{}'.format(count),            
load_best_model_at_end=True,     
evaluation_strategy="epoch")    


print("Training begins:\n")
trainer =  Trainer(
model=model,                         
args=training_args,                 
train_dataset=train_dataset,         
eval_dataset=valid_dataset,               
data_collator=data_collator,
callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

print(trainer.train())
print('---------------------')


## test data


test_data = pd.read_csv("policy_test_data.csv", "\t")
print("Encodings generation.")
test_encodings = tokenizer(test_data['Query'].tolist(),test_data['Segment'].tolist(), truncation=True, padding=True, max_length=max_length)
print("Encodings generated.")


test_dataset = Dataset(test_encodings)

# Make prediction

raw_pred, _, _ = trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)


import csv


def get_evaluation(test_file, X_pred):
    '''
    :param test_file:
    :param X_pred: List of binary relevance judgements for all sentences in all policies. 0 - Relevant, 1 - Irrelevant
    :return:
    '''

    X_test = []

    doc_ids = {}

    candidate_sentences = {}

    with open(test_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                print("Column names are " + str(row))
                line_count += 1
            else:

                X_test.append(row)
                DOCID = row[1]
                QUERYID = row[2]
                if DOCID not in doc_ids:
                    doc_ids[DOCID] = {}
                if QUERYID not in doc_ids[DOCID]:
                    doc_ids[DOCID][QUERYID] = []
                doc_ids[DOCID][QUERYID].append((row, line_count - 1))

                if DOCID not in candidate_sentences:
                    candidate_sentences[DOCID] = []
                candidate_sentences[DOCID].append(row[6])
                line_count += 1

    assert (len(X_pred) == len(X_test))

    full_test_score = np.array([0.0, 0.0, 0.0])
    full_total = 0

    for policy in doc_ids:
        for query in doc_ids[policy]:

            # Expert Annotations
            expert_relevant = {}

            for row_ind, row in enumerate(doc_ids[policy][query]):
                anns = row[0][-6:]
                for ann_ind, each_annotation in enumerate(anns):
                    if each_annotation != "None":
                        if ann_ind not in expert_relevant:
                            expert_relevant[ann_ind] = []

                    if each_annotation == "Relevant":
                        expert_relevant[ann_ind].append(row_ind)

            #print(expert_relevant)

            pred_relevant = []

            # Predicted
            for row_ind, row in enumerate(doc_ids[policy][query]):
                line_count = row[1]

                # 0 is relevant acording to the schema supplied to BERT
                if X_pred[line_count] == 0:
                    pred_relevant.append(row_ind)

            query_score = np.array([0.0, 0.0, 0.0])
            query_total = 0

            # Eval
            for each_ann in expert_relevant:

                other_ann = []

                for loo_ann in expert_relevant:
                    if loo_ann != each_ann:
                        other_ann.append(expert_relevant[loo_ann])

                all_reference_annotations = [
                    compute_f1(o_annotation, pred_relevant) for
                    o_annotation in other_ann]


                # We take the best
                best = max(all_reference_annotations, key=lambda x: x[2])

                query_score += best
                query_total += 1

            per_query_score = query_score / query_total

            full_test_score += per_query_score
            full_total += 1

    precision, recall = full_test_score[0] / full_total, full_test_score[1] / full_total
    f1 = (2 * precision * recall) / (precision + recall)
    print(precision)
    print(recall)
    print(f1)


test_path = "policy_test_data.csv"


print(get_evaluation(test_path, y_pred))
