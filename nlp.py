import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# 1. Device setup
device = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device == 0 else "Using CPU")

# 2. Load & preprocess the CSV
df = pd.read_csv(
    "/kaggle/input/expanded-equity-corpus4-csv/expanded_equity_corpus.csv"
)

df = df.dropna(subset=["Emotion"]).copy()
# Optionally drop pronoun-only entries
df = df[~df["Person"].isin(["she", "he", "him", "her"])].copy()
df["group_id"] = df["Race"]

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Emotion"]    # ← this makes sure each split has equal emotion proportions
)

# check counts
print(train_df["Emotion"].value_counts())
print(val_df  ["Emotion"].value_counts())

# ─── 3) HF DATASETS & TOKENIZATION ─────────────────────────────────────────────
import torch
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast, DataCollatorWithPadding

print(df.columns)
# 4.1 Create HF Datasets
hf_dsets = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "eval":  Dataset.from_pandas(val_df.reset_index(drop=True)),
})

# 4.2 Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 4.3 Preprocessing function
label2id = {"anger":0, "sadness":1, "fear":2, "joy":3}

# 3.4 Preprocessing fn
def preprocess(example):
    # tokenize
    toks = tokenizer(example["Sentence"], truncation=True, max_length=64)
    # map emotion → integer label
    toks["labels"] = label2id[example["Emotion"]]
    # carry group_id through (string race)
    toks["group_id"] = example["group_id"]
    return toks

# 3.5 Apply it, dropping all original columns by name (as a list)
original_cols = [
    "ID", "Sentence", "Template", "Person",
    "Gender", "Race", "Emotion", "Emotion word",
]
hf_dsets = hf_dsets.map(
    preprocess,
    remove_columns=original_cols,  # also drop the pandas index if present
    batched=False,
)
# 4.5 Data collator pads to the longest in the batch
data_collator = DataCollatorWithPadding(tokenizer)

# ─── 4) BUILD & TRAIN BINARY CLASSIFIER ────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

training_args = TrainingArguments(
    output_dir="runs/emotion_baseline",
    #evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dsets["train"],
    eval_dataset=hf_dsets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

from torch.nn import CrossEntropyLoss

# 6.1 Build your weight lookup, e.g. inverse-frequency:
# Suppose `counts` is a dict of dicts: counts[group][emotion_label] -> int
# And total = sum over all counts.
weights = {
    grp: {
        lbl: total_cnt / cnt 
        for lbl, cnt in emo_counts.items()
    }
    for grp, emo_counts in counts.items()
}

class ReweightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        group_ids = inputs.pop("group_id")  # strings, e.g. “Hispanic”
        
        outputs = model(**inputs)
        logits = outputs.logits
        
        # map string group → weight for each example
        example_weights = torch.tensor([
            weights[grp][lbl.item()]
            for grp, lbl in zip(group_ids, labels)
        ], device=logits.device)
        
        # per-example loss
        loss_fct = CrossEntropyLoss(reduction="none")
        per_example_loss = loss_fct(logits, labels)
        
        # weighted mean
        loss = (per_example_loss * example_weights).mean()
        
        return (loss, outputs) if return_outputs else loss

# 6.2 Instantiate and train
rew_trainer = ReweightedTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_dsets["train"],
    eval_dataset=hf_dsets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
rew_trainer.train()



# ─── After training ─────────────────────────────────────────────────────────────

# 1) Save the model weights + config
trainer.save_model("saved_models/sentiment_binary")

# 2) Save the tokenizer
tokenizer.save_pretrained("saved_models/sentiment_binary")


rew_trainer.save_model("saved_models/rew_sentiment_binary")
# ─── 5) INFERENCE: SENTIMENT-ANALYZE ALL SENTENCES ────────────────────────────

# Option A: with HuggingFace `Trainer.predict`
all_df   = df.reset_index(drop=True)             # original full set

# Option B: with a `pipeline` (more convenient for batches)
sent_pipe = pipeline(
    "sentiment-analysis",
    model=trainer.model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)
results = sent_pipe(all_df["Sentence"].tolist(), batch_size=64)
all_df["Predicted"] = [r["label"] for r in results]

# Now `all_df` contains your original columns + `Sentiment` (true) + `Predicted`.
# You can e.g.:
print(all_df[["Sentence","Sentiment","Predicted"]].head())
print("\nOverall accuracy:",
      (all_df["Sentiment"] == all_df["Predicted"]).mean())
