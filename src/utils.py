from transformers import BertTokenizer



class Config:
    NB_EPOCHS = 5
    LR = 3e-5
    EPS=1e-8
    MAX_LEN = 110
    N_SPLITS = 4
    BATCH_SIZE = 1
    TRAIN_BS = 60
    VALID_BS = 40
    MODEL_NAME ='bert-base-cased'
    TRAIN_FILE ='../input/train.csv'
    TOKENIZER =BertTokenizer.from_pretrained('bert-base-cased')