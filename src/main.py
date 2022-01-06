import torch
import torch.nn as nn
import pandas as pd
from utils import Config
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def __init__(self, questions1, questions2, labels):
        self.questions1 = questions1
        self.questions2 = questions2
        self.labels = labels
        self.tokenizer = Config.TOKENIZER

    def __len__(self):
        return len(self.questions1)

    def __getitem__(self, item):
        question1 = str(self.questions1[item])
        question2 = str(self.questions2[item])
        label = self.labels[item]
        question1_tokenized = self.tokenizer(
            question1,
            max_length=Config.MAX_LEN,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        question2_tokenized = self.tokenizer(
            question2,
            max_length=Config.MAX_LEN,
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


def similarity_ataloader(df, batch_size):
    """
    similarity_ataloader is a function that takes as input the dataset and returns a dataloader
    """
    dataset = SimilarityDataset(
        df["question1"].values, df["question2"].values, df["is_duplicate"].values)
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

    def __init__(self):
        super(BertForSimilarity, self).__init__()
        self.bert = BertModel.from_pretrained(Config.MODEL_NAME)
        self.maxpool = nn.MaxPool1d(3, stride=2)

    def forward(self, ids1, mask1, token1, ids2, mask2, token2):
        _, pooled_output = self.bert(
            ids1, attention_mask=mask1, token_type_ids=token1, return_dict=False)
        _, pooled_output2 = self.bert(
            ids2, attention_mask=mask2, token_type_ids=token2, return_dict=False)

        pooled_output_maxpooled = self.maxpool(pooled_output).detach().numpy()

        pooled_output2_maxpooled = self.maxpool(
            pooled_output2).detach().numpy()

        out = cosine_similarity(pooled_output_maxpooled.reshape(
            1, -1), pooled_output2_maxpooled.reshape(1, -1))
        return torch.tensor(out).flatten()


def train_model(model, dataloader, optimizer, criterion):
    """
    train_model is a function that takes as input the model, the dataloader, the optimizer and the criterion
    and trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        ids1 = batch['ids1'].to(device)
        mask1 = batch['mask1'].to(device)
        token1 = batch['token1'].to(device)
        ids2 = batch['ids2'].to(device)
        mask2 = batch['mask2'].to(device)
        token2 = batch['token2'].to(device)
        labels = batch['labels'].to(device)
        print(f"the labels {labels}")
        optimizer.zero_grad()
        outputs = model(ids1, mask1, token1, ids2, mask2, token2)
        print(f"the output {outputs}")
        loss = criterion(outputs.item(), labels.item())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        return total_loss / len(dataloader)


def evaluate_model(model, dataloader, criterion):
    """
    evaluate_model is a function that takes as input the model, the dataloader and the criterion
    and evaluates the model on the dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            ids1 = batch['ids1'].to(device)
            mask1 = batch['mask1'].to(device)
            token1 = batch['token1'].to(device)
            ids2 = batch['ids2'].to(device)
            mask2 = batch['mask2'].to(device)
            token2 = batch['token2'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(ids1, mask1, token1, ids2, mask2, token2)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    """
    main is a function that takes as input the model, the dataloader, the optimizer and the criterion
    and trains the model for one epoch.
    """
    model = BertForSimilarity().to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LR)
    criterion = nn.BCEWithLogitsLoss()

    df = pd.read_csv(Config.TRAIN_FILE).head(20)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataloader = similarity_ataloader(
        train_df, Config.BATCH_SIZE)
    valid_dataloader = similarity_ataloader(
        valid_df, Config.BATCH_SIZE)
    best_valid_loss = float('inf')

    for epoch in range(Config.NB_EPOCHS):

        train_loss = train_model(model, train_dataloader, optimizer, criterion)
        valid_loss = evaluate_model(model, valid_dataloader, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './model.pt')
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')


if __name__ == '__main__':
    main()
