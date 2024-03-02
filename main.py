import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import sklearn
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_report
from seqeval.scheme import IOB2
from tqdm import tqdm


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import dill as pkl
from argparse import Namespace, ArgumentParser
import gensim
from gensim.test.utils import datapath

import gc

os.system("wget http://vectors.nlpl.eu/repository/20/6.zip")
os.system("unzip 6.zip")
os.system("unzip 6.zip -d wikipedia")

word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format('wikipedia/model.txt')

class VocabularyEmbedding(object):
    def __init__(self, gensim_w2v):

        self.w2v = gensim_w2v
        self.w2v.add_vector('<s>', np.random.uniform(low=-1, high=1.0, size=(300,)))
        self.w2v.add_vector('</s>', np.random.uniform(low=-1, high=1.0, size=(300,)))
        self.w2v.add_vector('<pad>', np.random.uniform(low=-1, high=1.0, size=(300,)))
        self.w2v.add_vector('<unk>', np.random.uniform(low=-1, high=1.0, size=(300,)))

        bos = self.w2v.key_to_index.get('<s>')
        eos = self.w2v.key_to_index.get('</s>')
        pad = self.w2v.key_to_index.get('<pad>')
        unk = self.w2v.key_to_index.get('<unk>')

        self.bos_index = bos
        self.eos_index = eos
        self.pad_index = pad
        self.unk_index = unk


    def tokenizer(self, text):
        return [t for t in text.split(' ')]

    def encode(self, text):

        sequence = []

        tokens = self.tokenizer(text)
        for token in tokens:

            index = self.w2v.key_to_index.get(token, self.unk_index)
            sequence.append(index)

        return sequence

    def create_padded_tensor(self, sequences):

        lengths = [len(sequence) for sequence in sequences]
        max_seq_len = max(lengths)
        tensor = torch.full((len(sequences), max_seq_len), self.pad_index, dtype=torch.long)

        for i, sequence in enumerate(sequences):
            for j, token in enumerate(sequence):
                tensor[i][j] = token

        return tensor, lengths
    

class BIOTagSequencer(object):
    def __init__(self, tag_corpus, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>'):
        self.word2idx = {}
        self.idx2word = {}
        self.unk_index = self.add_token(unk_token)
        self.pad_index = self.add_token(pad_token)
        self.bos_index = self.add_token(bos_token)
        self.eos_index = self.add_token(eos_token)
        self.tokenizer = lambda text: [t for t in text.split(' ')]

        for _tags in tag_corpus:
          for _token in self.tokenizer(_tags):
            self.add_token(_token)

    def add_token(self, token):
        if token not in self.word2idx:
          self.word2idx[token] = new_index = len(self.word2idx)
          self.idx2word[new_index] = token
          return new_index

        else:
          return self.word2idx[token]

    def encode(self, text):
        tokens = self.tokenizer(text)

        sequence = []

        for token in tokens:

            index = self.word2idx.get(token, self.unk_index)
            sequence.append(index)

        return sequence

    def create_padded_tensor(self, sequences):

        lengths = [len(sequence) for sequence in sequences]
        max_seq_len = max(lengths)
        tensor = torch.full((len(sequences), max_seq_len), self.pad_index, dtype=torch.long)

        for i, sequence in enumerate(sequences):
            for j, token in enumerate(sequence):
                tensor[i][j] = token

        return tensor, lengths
    


class CoreRelationSequencer:
    def __init__(self, relation_corpus):
        self.mlb = MultiLabelBinarizer()
        relations = [set(relation.split()) for relation in relation_corpus]
        self.mlb.fit(relations)

    def encode(self, relation_string):
        relations = set(relation_string.split())
        encoded = self.mlb.transform([relations])
        return encoded[0]

    def decode(self, encoded_vector):
        relations = self.mlb.inverse_transform(np.array([encoded_vector]))
        return ' '.join(relations[0])

    def output_size(self):
        """Return the number of unique core relations (the size of the one-hot encoded vector)."""
        return len(self.mlb.classes_)

    def create_padded_tensor(self, sequences):
        max_seq_len = max(len(seq) for seq in sequences)
        num_classes = len(self.mlb.classes_)

        # Initialize a tensor of zeros
        tensor = torch.zeros((len(sequences), max_seq_len, num_classes), dtype=torch.float)

        for i, sequence in enumerate(sequences):
            encoded_seq = self.encode(sequence)
            seq_len = len(encoded_seq)

            # Assign the one-hot encoded sequence to the tensor
            tensor[i, :seq_len, :] = torch.tensor(encoded_seq, dtype=torch.float)

        return tensor
    


class TaggerDataset(Dataset):
    def __init__(self, data, text_sequencer, core_sequencer, bio_sequencer):

        self.data = data

        self.input_sequencer = text_sequencer

        self.core_sequencer = core_sequencer
        self.bio_sequencer = bio_sequencer

    def __getitem__(self, index):
        text, core_tags, bio_tags = self.data[index]

        x = self.input_sequencer.encode(text)
        y = self.core_sequencer.encode(core_tags)
        z = self.bio_sequencer.encode(bio_tags)

        return x, y, z

    def __len__(self):
        return len(self.data)



class Transformer_Model(nn.Module):
    def __init__(self, core_output_size, bio_output_size, pad_index, d_model=300, n_heads=4, d_hidden=256, n_layers=2, dropout_p=0.1, w2v_weights=word2vec_weights):
        super(Transformer_Model, self).__init__()
        self.pad_index = pad_index

        # tunable embeddings
        self.embedding = nn.Embedding.from_pretrained(w2v_weights, freeze=False)

        # frozen embeddings
        self.frozen_embedding = nn.Embedding.from_pretrained(w2v_weights, freeze=True)

        self.dropout = nn.Dropout(dropout_p)

        # Define the transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_hidden,
            dropout=dropout_p
        )

        # Stack multiple layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final linear layer
        self.core_fc = nn.Linear(d_model, core_output_size, bias=False)
        self.bio_fc = nn.Linear(d_model, bio_output_size, bias=False)


    def forward(self, x, lengths):
        # Create a mask for padding
        pad_mask = (x == self.pad_index)#.transpose(0, 1)

        # Embeddings
        embed = self.embedding(x)
        embed_frozen = self.frozen_embedding(x)

        # Combine embeddings
        #embed_combined = self.dropout(torch.cat((embed, embed_frozen), dim=2))
        embed_combined = self.dropout(embed)

        # Adjust dimensions for Transformer
        embed_combined = embed_combined.permute(1, 0, 2)  # Shape: [seq_len, batch_size, d_model]

        # Transformer Encoder
        transformer_output = self.transformer_encoder(embed_combined, src_key_padding_mask=pad_mask)

        # Permute back to [batch_size, seq_len, d_model]
        transformer_output = transformer_output.permute(1, 0, 2)

        # Apply final linear layers for core and bio
        core_logits = self.core_fc(transformer_output).mean(dim=1)
        bio_logits = self.bio_fc(transformer_output)

        return core_logits, bio_logits
    

def prepare_batch(batch, in_sequencer, core_sequencer, bio_sequencer):
    texts, core_labels, bio_labels = zip(*batch)
    text_tensor, lengths = in_sequencer.create_padded_tensor(texts)

    # Check if core_labels need encoding
    if isinstance(core_labels[0], str):
        core_tensor = core_sequencer.create_padded_tensor(core_labels)
    else:
        # If already one-hot encoded, just convert to tensor
        core_tensor = torch.tensor(core_labels, dtype=torch.float)

    bio_tensor, _ = bio_sequencer.create_padded_tensor(bio_labels)
    #print("Core tensor shape:", core_tensor.shape)
    #print("Bio tensor shape:", bio_tensor.shape)
    return (text_tensor, lengths, core_tensor, bio_tensor)



def train(model, optimizer, core_loss_function, bio_loss_function, loader, core_labels, bio_labels, log_every_n=100):
    model.train()

    total_loss = 0
    all_core_preds = []
    all_core_true = []
    all_bio_preds = []
    all_bio_true = []

    for i, batch in enumerate(tqdm(loader, desc="Training")):
        optimizer.zero_grad()

        # Get inputs and targets from the batch
        inputs, lengths, core_targets, bio_targets = batch

        # Forward pass: Get core and bio logits from the model
        core_logits, bio_logits = model(inputs, lengths)
        core_targets = core_targets.view_as(core_logits)

        # Compute loss for each set of logits
        core_loss = core_loss_function(core_logits, core_targets)
        bio_loss = bio_loss_function(bio_logits.view(-1, bio_logits.size(-1)), bio_targets.view(-1))

        # Combine losses
        combined_loss = core_loss + bio_loss
        total_loss += combined_loss.item()

        # Backward pass and optimization
        combined_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        # Collect predictions and true labels for metrics calculation
        all_core_preds.extend(core_logits.sigmoid().round().detach().cpu().numpy())
        all_core_true.extend(core_targets.detach().cpu().numpy())

        bio_preds = torch.argmax(bio_logits, -1).detach().cpu().numpy()
        bio_true = bio_targets.detach().cpu().numpy()
        for bp, bt in zip(bio_preds, bio_true):
            all_bio_preds.append([bio_labels[label] for label in bp])
            all_bio_true.append([bio_labels[label] for label in bt])

        # Optionally log the loss
        if log_every_n > 0 and (i + 1) % log_every_n == 0:
            print(f"Batch {i + 1}: Combined Loss = {total_loss / (i + 1)}")

    # Flatten core predictions and true labels for classification report
    flat_core_preds = np.argmax(all_core_preds, axis=1)
    flat_core_true = np.argmax(all_core_true, axis=1)

    # Print classification report for core relations
    print("Core Relation Classification Report:")
    print(classification_report(flat_core_true, flat_core_preds, target_names=core_labels))

    # Print classification report for BIO tagging
    print("BIO Tagging Report:")
    print(seqeval_report(all_bio_true, all_bio_preds, scheme=IOB2))

    return combined_loss, (total_loss / len(loader))



def run_training(model, optimizer, core_loss_function, bio_loss_function, train_loader, valid_loader, core_labels, bio_labels, n_epochs=20):
        train_losses = []
        train_running_losses = []

        valid_losses = []
        valid_running_losses = []

        for i in range(n_epochs):
            print(f"Epoch: {i}")
            train_loss_history, train_running_loss_history = train(model, optimizer, core_loss_function, bio_loss_function, train_loader, core_labels, bio_labels, log_every_n=100)
            train_losses.append(train_loss_history)
            train_running_losses.append(train_running_loss_history)

            # val_loss_history, val_running_loss_history = train(model, optimizer, core_loss_function, bio_loss_function, valid_loader, core_labels, bio_labels, log_every_n=100)
            # valid_losses.append(val_loss_history)
            # valid_running_losses.append(val_running_loss_history)

        #all_train_losses = list(chain.from_iterable(train_losses))
        #all_train_running_losses = list(chain.from_iterable(train_running_losses))

        #train_epoch_idx = range(len(all_train_running_losses))


        return train_losses, train_running_losses, valid_losses, valid_running_losses


def evaluate(model, loader, core_labels, bio_labels):
    model.eval()  # Set the model to evaluation mode

    all_core_preds = []
    all_core_true = []
    all_bio_preds = []
    all_bio_true = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(loader, desc="Evaluating"):
            inputs, lengths, core_targets, bio_targets = batch

            core_logits, bio_logits = model(inputs, lengths)
            core_targets = core_targets.view_as(core_logits)

            all_core_preds.extend(core_logits.sigmoid().round().cpu().numpy())
            all_core_true.extend(core_targets.cpu().numpy())

            bio_preds = torch.argmax(bio_logits, -1).cpu().numpy()
            bio_true = bio_targets.cpu().numpy()
            for bp, bt in zip(bio_preds, bio_true):
                all_bio_preds.append([bio_labels[label] for label in bp])
                all_bio_true.append([bio_labels[label] for label in bt])

    # Flatten core predictions and true labels for classification report
    mlb = MultiLabelBinarizer(classes=range(len(core_labels)))
    flat_core_preds = np.argmax(all_core_preds, axis=1)
    flat_core_true = np.argmax(all_core_true, axis=1)

    present_classes = np.unique(np.concatenate([np.argmax(all_core_true, axis=1),
                                                np.argmax(all_core_preds, axis=1)]))
    #print(present_classes)

    # Print classification report for core relations
    print("Core Relation Classification Report:")
    print(classification_report(flat_core_true, flat_core_preds,
                                target_names=[core_labels[i] for i in present_classes]))

    # Print classification report for BIO tagging
    print("BIO Tagging Report:")
    print(seqeval_report(all_bio_true, all_bio_preds, scheme=IOB2))



def make_predictions(dataset, model, text_sequencer, core_sequencer, bio_sequencer, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    for data in DataLoader(dataset, batch_size=1, shuffle=False):
        text, core_relation, bio_tag = data
        # Ensure text is in tensor format
        text_tensor, lengths = text_sequencer.create_padded_tensor([text[0]])
        text_tensor = text_tensor.to(device)
        lengths = torch.tensor(lengths, dtype=torch.long).to(device)

        with torch.no_grad():
            core_logits, bio_logits = model(text_tensor, lengths)
            core_predicted = core_logits.sigmoid().round().cpu().numpy()[0]
            bio_predicted_indices = bio_logits.argmax(dim=2).cpu().numpy()[0]

            # Convert predicted indices to tags
            bio_predicted_tags = [bio_sequencer.idx2word[idx] for idx in bio_predicted_indices[:lengths[0]]]
            predictions.append((core_predicted, bio_predicted_tags))

    return predictions



def main():
    parser = ArgumentParser("example")

    parser.add_argument('--train', action="store_true", help="indicator to train model")
    parser.add_argument('--test', action="store_true", help="indicator to test model")

    parser.add_argument('--data', help="path to data file")
    parser.add_argument('--save_model', help="ouput path of trained model")
    parser.add_argument('--model_path', help="path to load trained model from")

    parser.add_argument('--output', help="output path of predictions")

    args = parser.parse_args()

    word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format('wikipedia/model.txt')



    if args.train:
        print('training model')

        train_data_path = args.data

        train_data = pd.read_csv(train_data_path)

        df = pd.read_csv(train_data_path)
        df['IOB Slot tags'] = df['IOB Slot tags'].replace(np.nan, "none")

        item_list = list(row.to_dict() for idx, row in df.iterrows())
        np.random.shuffle(item_list)
        item_list = [item for item in item_list if len(item['utterances'].split(' ')) == len(item['IOB Slot tags'].split(' '))]

        train_df, validation_df = train_test_split(item_list, test_size=0.1, train_size=0.9)

        # Convert train_df and validation_df back to DataFrames
        train_df = pd.DataFrame(train_df)
        validation_df = pd.DataFrame(validation_df)

        # Replace NaN values in the Core Relations column with a placeholder string
        train_df['Core Relations'] = train_df['Core Relations'].fillna("none")
        validation_df['Core Relations'] = validation_df['Core Relations'].fillna("none")

        relation_corpus = [row['Core Relations'] for index, row in train_df.iterrows()]

        train_data = [(row['utterances'], row['Core Relations'], row['IOB Slot tags']) for index, row in train_df.iterrows()]
        val_data = [(row['utterances'], row['Core Relations'], row['IOB Slot tags']) for index, row in validation_df.iterrows()]

        train_texts = list([e[0] for e in train_data])
        train_core_tags = list([e[1] for e in train_data])
        train_bio_tags = list([e[2] for e in train_data])

        text_sequencer = VocabularyEmbedding(word2vec_weights)
        core_sequencer = CoreRelationSequencer(relation_corpus)
        bio_sequencer = BIOTagSequencer(train_bio_tags)

        with open('core_sequencer.pkl', 'wb') as f:
            pkl.dump(core_sequencer, f)

        with open('bio_sequencer.pkl', 'wb') as f:
            pkl.dump(bio_sequencer, f)


        train_dataset = TaggerDataset(train_data, text_sequencer, core_sequencer, bio_sequencer)
        val_dataset = TaggerDataset(val_data, text_sequencer, core_sequencer, bio_sequencer)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=lambda batch: prepare_batch(batch, text_sequencer, core_sequencer, bio_sequencer))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, collate_fn=lambda batch: prepare_batch(batch, text_sequencer, core_sequencer, bio_sequencer), shuffle=False)

        hidden_size=10
        core_output_size = core_sequencer.output_size()
        bio_output_size = len(bio_sequencer.idx2word)
        pad_index = bio_sequencer.pad_index

        #print("Core output size:", core_output_size)
        #print("BIO output size:", bio_output_size)

        w2v_weights_array = word2vec_weights.vectors
        w2v_weights_tensor = torch.tensor(w2v_weights_array, dtype=torch.float)


        model = Transformer_Model(core_output_size, bio_output_size, pad_index, w2v_weights=w2v_weights_tensor)


        learning_rate = 1e-3
        core_loss_function = nn.BCEWithLogitsLoss() # Avoid computing loss on padding tokens
        bio_loss_function = nn.CrossEntropyLoss(ignore_index=bio_sequencer.pad_index) # Avoid computing loss on padding tokens
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        core_label_list = list(core_sequencer.mlb.classes_)
        bio_label_list = [bio_sequencer.idx2word[i] for i in range(len(bio_sequencer.idx2word))]
        train_losses, train_running_losses, valid_losses, valid_running_losses = run_training(model, optimizer, core_loss_function, 
                                                                                              bio_loss_function, train_loader, val_loader, 
                                                                                              core_label_list, bio_label_list, n_epochs=20)
        
        evaluate(model, val_loader, core_label_list, bio_label_list)


    if args.test:
        print('testing model')
        # put code to run testing here
        test_data_path = args.data
        #test_data_path = 'hw1_test.csv'
        test_df = pd.read_csv(test_data_path)
        test_data = [(i['utterances']) for _, i in test_df.iterrows()]

        text_sequencer = VocabularyEmbedding(word2vec_weights)
        with open('core_sequencer.pkl', 'rb') as f:
            core_sequencer = pkl.load(f)

        with open('bio_sequencer.pkl', 'rb') as f:
            bio_sequencer = pkl.load(f)

        hidden_size=10
        core_output_size = core_sequencer.output_size()
        bio_output_size = len(bio_sequencer.idx2word)
        pad_index = bio_sequencer.pad_index

        #print("Core output size:", core_output_size)
        #print("BIO output size:", bio_output_size)

        w2v_weights_array = word2vec_weights.vectors
        w2v_weights_tensor = torch.tensor(w2v_weights_array, dtype=torch.float)


        model = Transformer_Model(core_output_size, bio_output_size, pad_index, w2v_weights=w2v_weights_tensor)


        learning_rate = 1e-3
        core_loss_function = nn.BCEWithLogitsLoss() # Avoid computing loss on padding tokens
        bio_loss_function = nn.CrossEntropyLoss(ignore_index=bio_sequencer.pad_index) # Avoid computing loss on padding tokens
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        core_label_list = list(core_sequencer.mlb.classes_)
        bio_label_list = [bio_sequencer.idx2word[i] for i in range(len(bio_sequencer.idx2word))]


        dummy_core_relations = ['none' for _ in test_data]  # Replace 'none' with appropriate dummy relation
        dummy_iob_tags = ['O ' * len(utterance.split()) for utterance in test_data]  # 'O' tag for each word

        # Pair each utterance with its corresponding dummy tags
        test_data = list(zip(test_data, dummy_core_relations, dummy_iob_tags))

        # Create the test dataset using TaggerDataset
        test_dataset = TaggerDataset(test_data, text_sequencer, core_sequencer, bio_sequencer)

        # Create DataLoader for the test dataset
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, collate_fn=lambda 
                                                  batch: prepare_batch(batch, text_sequencer, core_sequencer, bio_sequencer))
        
        test_predictions = make_predictions(test_dataset, model, text_sequencer, core_sequencer, bio_sequencer)

        core_relation_names = core_sequencer.mlb.classes_

        def format_core_relations(core_relations):
            relations = [core_relation_names[i] for i, val in enumerate(core_relations) if val == 1]
            return ' '.join(relations) if relations else 'none'

        core_relation_predictions = [format_core_relations(core_pred) for core_pred, _ in test_predictions]
        bio_tag_predictions = [' '.join(tags) for _, tags in test_predictions]

        predicted_tags_df = pd.DataFrame({
            'utterances': test_df['utterances'],
            'IOB Slot tags': bio_tag_predictions,
            'Core Relations': core_relation_predictions
        })

        predicted_tags_df.to_csv('preds.csv', index=False)

        predictions_df = pd.read_csv('preds.csv')

        def ensure_tags(utterance, tags):
            tag_list = tags.split()
            word_list = utterance.split()

            # If the number of tags is less than the number of words, pad with 'O'
            if len(tag_list) < len(word_list):
                tag_list.extend(['O'] * (len(word_list) - len(tag_list)))

            return ' '.join(tag_list)

        predictions_df['IOB Slot tags'] = predictions_df.apply(
            lambda row: ensure_tags(row['utterances'], row['IOB Slot tags']), axis=1
        )

        predictions_df.to_csv('predictions.csv', index=False)


if __name__ == "__main__":
    main()