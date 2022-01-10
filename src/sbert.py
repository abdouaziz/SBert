import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Pretrained Machine Translation French to Wolof")

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv  file containing the training data."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=150,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total number of training steps to perform the model .",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--epsilone",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-cased",
        help="Pretrained model name.",
    )

    args = parser.parse_args()

    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in [
            "csv", "json"], "`train_file` should be a csv or a json file."

    if args.model_name is not None:

        assert args.model_name in ["bert-base-cased", "bert-base-uncased", "bert-large-cased",
                                   "bert-large-uncased", "roberta"], "`model_name` should be bert or roberta ."

    return args


class SimilarityDataset(Dataset):
    """
    SimilarityDataset is a subclass of Dataset.
    It is used to create a dataset for the model.
    Take as input two questions , i.e questions1 and questions2 and their labels (0 or 1) for their similarity.
    question1 and question2 are tokenized and padded to the same length.
    questions1 : list of strings
    questions2 : list of strings
    labels : list of ints
    return the dataset as a dictionary , input_ids , attention_mask , token_type_ids , labels
    """

    def __init__(self, questions1, questions2, labels, tokenizer, max_length):
        self.questions1 = questions1
        self.questions2 = questions2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions1)

    def __getitem__(self, item):
        question1 = str(self.questions1[item])
        question2 = str(self.questions2[item])
        label = self.labels[item]
        question1_tokenized = self.tokenizer(
            question1,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        question2_tokenized = self.tokenizer(
            question2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'

        )

        return {
            'ids1': question1_tokenized["input_ids"].flatten(),
            'mask1': question1_tokenized["attention_mask"].flatten(),
            'token1': question1_tokenized["token_type_ids"].flatten(),
            'ids2': question2_tokenized["input_ids"].flatten(),
            'mask2': question2_tokenized["attention_mask"].flatten(),
            'token2': question2_tokenized["token_type_ids"].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


def similarity_dataloader(df, batch_size, tokenizer, max_length):
    """
    similarity_dataloader is a function that takes as input the dataset and returns a dataloader
    """
    dataset = SimilarityDataset(
        df["question1"].values, df["question2"].values, df["is_duplicate"].values, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    return dataloader


class BertForSimilarity(nn.Module):
    """
    BertForSimilarity is a class that inherits from nn.Module.
    It is used to create a model for the model.
    return the cosinus similarity between the two questions
    """

    def __init__(self, device):
        super(BertForSimilarity, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.device = device
        self.mean_pool = nn.AvgPool1d(3, stride=2)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(1149, 2)

    def forward(self, ids1, mask1, token1, ids2, mask2, token2):
        _, pooled_output = self.bert(
            ids1, attention_mask=mask1, token_type_ids=token1, return_dict=False)
        _, pooled_output2 = self.bert(
            ids2, attention_mask=mask2, token_type_ids=token2, return_dict=False)

        u = self.mean_pool(pooled_output).cpu().detach().numpy()

        v = self.mean_pool(pooled_output2).cpu().detach().numpy()

        u_v = torch.abs(torch.tensor(u-v)).cpu().detach().numpy()

        concat = torch.concat((torch.tensor(u).to(self.device), torch.tensor(
            v).to(self.device), torch.tensor(u_v).to(self.device)))
        concat = concat.flatten()

        out = self.out(concat)

        return out


def loss_fn(outputs, labels):

    return nn.CrossEntropyLoss()(outputs, labels)


def yield_optimizer(model, args):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_parameters, lr=args.learning_rate, eps=args.episilone)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):

        ids1 = data['ids1'].to(device)
        mask1 = data['mask1'].to(device)
        token1 = data['token1'].to(device)
        ids2 = data['ids2'].to(device)
        mask2 = data['mask2'].to(device)
        token2 = data['token2'].to(device)
        targets = data['labels'].to(device)
      #  optimizer.zero_grad()
        outputs = model(ids1, mask1, token1, ids2, mask2, token2)

        _, pred = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(pred == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):

    model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():

        for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):

            ids1 = data['ids1'].to(device)
            mask1 = data['mask1'].to(device)
            token1 = data['token1'].to(device)
            ids2 = data['ids2'].to(device)
            mask2 = data['mask2'].to(device)
            token2 = data['token2'].to(device)
            targets = data['labels'].to(device)
        #  optimizer.zero_grad()
            outputs = model(ids1, mask1, token1, ids2, mask2, token2)

            _, pred = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(pred == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def main():

    args = parse_args()
    model = BertForSimilarity(device)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    df = pd.read_csv(args.data_path)

    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42)

    traindataloader = similarity_dataloader(
        train_data, args.batch_size, tokenizer, args.max_length)
    testdataloader = similarity_dataloader(
        test_data, args.batch_size, tokenizer, args.max_length)

    best_accuracy = 0
    some_val = 0

    nb_train_steps = int(len(traindataloader) /
                         args.train_batch_size * args.epochs)
    optimizer = yield_optimizer(model)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=nb_train_steps)

    for epoch in range(args.epochs):

        print(f'Epoch {epoch + 1}')

        train_acc, train_loss = train_epoch(
            model, traindataloader, loss_fn, optimizer, device, scheduler, len(train_data))

        print(f"Train accuracy {train_acc} ,Train Loss {train_loss}")

        val_acc, val_loss = eval_model(
            model, testdataloader, loss_fn, device, len(test_data))

        print(f"Validation accuracy {val_acc} , Validation loss {val_loss}")

        if val_acc > best_accuracy:

            torch.save(model.state_dict(), 'best_model.bin')
            best_accuracy = val_acc
            print(f"Best accuracy {best_accuracy}")


if __name__ == '__main__':
    main()
