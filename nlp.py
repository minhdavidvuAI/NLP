import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from torch.nn import CrossEntropyLoss
import re

########## Data exploration ############
device = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device == 0 else "Using CPU")

df = pd.read_csv("/kaggle/input/expanded-equity-corpus4-csv/expanded_equity_corpus.csv")

# Display the first few rows
print(df.head(2))

print()
# Person names
print(df["Person"].value_counts())
print()
# Emotion labels
print(df["Emotion"].value_counts())
print()

### first check of base BERT model
# Latisha gets more text labeled as negative, than names like sebastian
sentiment_pipeline = pipeline("sentiment-analysis", device=device)
df["Sentiment"] = df["Sentence"].apply(lambda x: sentiment_pipeline(x)[0]['label'])

sentiment_counts = df.groupby(["Person", "Sentiment"]).size().unstack(fill_value=0)
sentiment_counts = sentiment_counts.sort_values(by="NEGATIVE", ascending=False)

print("### Sentiment analysis ###")
print(sentiment_counts)

# check person sebastian
df2 = df.copy()
df2 = df2[df2["Person"] == "Sebastian"][["Sentence", "Emotion", "Sentiment"]]

test = df2.groupby(["Emotion", "Sentiment"]).size().unstack(fill_value=0)

test = test.sort_values(by="NEGATIVE", ascending=False)
print("\nPerson: Sebastian")
print(test)

# check person latisha
df3 = df.copy()
df3 = df3[df3["Person"] == "Latisha"][["Sentence", "Emotion", "Sentiment"]]

test = df3.groupby(["Emotion", "Sentiment"]).size().unstack(fill_value=0)

test = test.sort_values(by="NEGATIVE", ascending=False)
print("\nPerson: Latisha")
print(test)

# race sentiment analysis
# race (names) get more often labeled as NEGATIVE than other races (names)
print("\n### Race Sentiment analysis ###")
df4 = df.copy()
df4 = df4[["Race", "Emotion", "Sentiment"]]

test = df4.groupby(["Race","Emotion", "Sentiment"]).size().unstack(fill_value=0)

test = test.sort_values(by=["Race","Emotion"], ascending=False)
print(test)




########## Model finetuning ############
print("\n")
print("Start model finetuning")

CSV_PATH = "expanded_equity_corpus.csv"
SENTIMENT_LABELS = ["NEGATIVE", "POSITIVE"]
label2id = {lab: i for i, lab in enumerate(SENTIMENT_LABELS)}
id2label = {i: lab for lab, i in label2id.items()}

device_id = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device_id >= 0 else "Using CPU")


# Load data
df = pd.read_csv(CSV_PATH).dropna(subset=["Emotion"])
#df = df[~df["Person"].isin(["she", "he", "him", "her"])].copy()

df["group_id"] = df["Race"]
#create col for ground truth
df["sentiment"] = df["Emotion"].apply(lambda x: "POSITIVE" if x == "joy" else "NEGATIVE") 

# get some test data - not included in training!
test_persons = ["Alia", "Diego", "Jasmine"]
test_df = df[df["Person"].isin(test_persons)].copy()

# Create remaining data for training/validation
df = df[~df["Person"].isin(test_persons)].copy()

# make equal splits by Emotion (joy, sadness, anger, fear)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Emotion"]
)

# sanity check
print("Train emotion counts:\n", train_df["Emotion"].value_counts())
print("Val   emotion counts:\n", val_df["Emotion"].value_counts())


#tokenize 
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id, max_length=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        toks = self.tokenizer(
            row["Sentence"],
            truncation=True,
            padding=False,         # padding done by DataCollator
            max_length=self.max_length,
        )
        return {
            "input_ids":      torch.tensor(toks["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(toks["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(self.label2id[row["sentiment"]], dtype=torch.long),
            "group_id":       row["group_id"],  # passed through for reweighting
        }

train_ds = SentimentDataset(train_df, tokenizer, label2id)
eval_ds  = SentimentDataset(val_df,  tokenizer, label2id)

data_collator = DataCollatorWithPadding(tokenizer)


# get base model (BERT)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(SENTIMENT_LABELS),
    id2label=id2label,
    label2id=label2id,
)

# define training args
training_args = TrainingArguments(
    output_dir="runs/emotion_baseline",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=50,
)

# create trainer model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Start fine tuning...")
trainer.train()

print("Fine tuning finished. Saving models in saved_models/baseline_model")
trainer.save_model("saved_models/baseline_model")
tokenizer.save_pretrained("saved_models/emotion_baseline_tokenizer")

print("Saving complete.")

# Check accuracy after fine tuning
print("Accuracy check...")
all_df = df.reset_index(drop=True)

sent_pipe = pipeline(
    "sentiment-analysis",
    model=trainer.model,
    tokenizer=tokenizer,
    device=device_id,
)

results = sent_pipe(all_df["Sentence"].tolist(), batch_size=64)
all_df["Predicted_Sentiment"] = [r["label"] for r in results]

# Accuracy check on train data
# of course fitted data so acc should be 100%
print("\nOverall sentiment accuracy:", (all_df["sentiment"] == all_df["Predicted_Sentiment"]).mean())
train_acc = all_df[["Race", "Emotion", "Predicted_Sentiment"]].rename(columns={"Predicted_Sentiment": "Sentiment"})
test = train_acc.groupby(["Race", "Emotion", "Sentiment"]).size().unstack(fill_value=0)
test = test.sort_values(by=["Race", "Emotion"], ascending=[True, True])

print(test)

# Accuracy check on test set
print("Testset accuracy...")
results = sent_pipe(test_df["Sentence"].tolist(), batch_size=64)
test_df["Predicted_Sentiment"] = [r["label"] for r in results]

test_acc = test_df[["Race", "Emotion", "Predicted_Sentiment"]].rename(columns={"Predicted_Sentiment": "Sentiment"})
test = test_acc.groupby(["Race", "Emotion", "Sentiment"]).size().unstack(fill_value=0)
test = test.sort_values(by=["Race", "Emotion"], ascending=[True, True])

print(test)



##### Final test for fine tuned model on Twitter data #####

# Load the data
df = pd.read_csv("final_sentiment.csv")

# search some names 
keywords = ["latisha", "diego", "sebastian", "my mother"]
pattern = re.compile(r"|".join(keywords), re.IGNORECASE)

# Filter rows where 'text' contains any of the keywords
filtered_df = df[df["text"].apply(lambda x: isinstance(x, str) and bool(pattern.search(x)))]

differences = filtered_df[filtered_df["Sentiment_fine"] != filtered_df["Predicted_Sentiment"]]
print(f"Number of differing predictions: {differences.shape[0]}")


print("\n")
for idx, row in differences[["text","Predicted_Sentiment","Sentiment_fine"]].iterrows():
    print(f"\nRow {idx}")
    print("Text:", row["text"])
    print("Prediction:", row["Predicted_Sentiment"])
    print("Fine tuned:", row["Sentiment_fine"])
