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

# ─── 1) CONFIG & DEVICE ───────────────────────────────────────────────────────
CSV_PATH = "expanded_equity_corpus.csv"
EMOTION_LABELS = ["anger", "sadness", "fear", "joy"]
label2id = {lab: i for i, lab in enumerate(EMOTION_LABELS)}
id2label = {i: lab for lab, i in label2id.items()}

device_id = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device_id >= 0 else "Using CPU")


# ─── 2) LOAD & SPLIT ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).dropna(subset=["Emotion"])
df = df[~df["Person"].isin(["she", "he", "him", "her"])].copy()
df["group_id"] = df["Race"]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Emotion"]
)

print("Train emotion counts:\n", train_df["Emotion"].value_counts())
print("Val   emotion counts:\n", val_df["Emotion"].value_counts())


# ─── 3) TOKENIZER & DATASET ───────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

class EmotionDataset(Dataset):
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
            "labels":         torch.tensor(self.label2id[row["Emotion"]], dtype=torch.long),
            "group_id":       row["group_id"],  # passed through for reweighting
        }

train_ds = EmotionDataset(train_df, tokenizer, label2id)
eval_ds  = EmotionDataset(val_df,  tokenizer, label2id)

data_collator = DataCollatorWithPadding(tokenizer)


# ─── 4) BASELINE TRAINER ──────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("=== Starting baseline training ===")
trainer.train()

trainer.save_model("saved_models/emotion_baseline")
tokenizer.save_pretrained("saved_models/emotion_baseline")

print("=== Done ===")

# ─── 8) INFERENCE ON ALL SENTENCES ────────────────────────────────────────────
all_df = df.reset_index(drop=True)

sent_pipe = pipeline(
    "sentiment-analysis",
    model=trainer.model,
    tokenizer=tokenizer,
    device=device_id,
)

results = sent_pipe(all_df["Sentence"].tolist(), batch_size=64)
all_df["Predicted"] = [r["label"] for r in results]

print(all_df[["Sentence", "Emotion", "Predicted"]])
print("\nOverall accuracy:",
      (all_df["Emotion"] == all_df["Predicted"]).mean())

# ─── 5) COMPUTE GROUP×EMOTION WEIGHTS ─────────────────────────────────────────
# count occurrences in training set
count_series = train_df.groupby(["group_id", "Emotion"]).size()
total = count_series.sum()
# build nested dict: weights[group][label] = total/count
weights = {
    grp: {
        label2id[emo]: total / cnt
        for (g, emo), cnt in count_series.items() if g == grp
    }
    for grp in train_df["group_id"].unique()
}


# ─── 6) REWEIGHTED TRAINER ────────────────────────────────────────────────────
class ReweightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        group_ids = inputs.pop("group_id")   # list of strings
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # map each example → its weight
        example_weights = torch.tensor(
            [weights[grp][lbl.item()] for grp, lbl in zip(group_ids, labels)],
            device=logits.device,
        )
        
        per_example_loss = CrossEntropyLoss(reduction="none")(logits, labels)
        loss = (per_example_loss * example_weights).mean()
        
        return (loss, outputs) if return_outputs else loss

rew_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(EMOTION_LABELS),
    id2label=id2label,
    label2id=label2id,
)

rew_trainer = ReweightedTrainer(
    model=rew_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("=== Starting reweighted training ===")
rew_trainer.train()


# ─── 7) SAVE ARTIFACTS ────────────────────────────────────────────────────────

rew_trainer.save_model("saved_models/emotion_reweighted")


