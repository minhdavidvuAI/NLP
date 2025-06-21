import pandas as pd
import torch
from transformers import pipeline
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, pipeline

df = pd.read_csv("Tweets.csv")

device = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device == 0 else "Using CPU")
# Load sentiment pipeline (assuming CUDA device ID is already defined as 'device')
sentiment_pipeline = pipeline("sentiment-analysis", device=device)

# Apply sentiment analysis to the 'text' column
#df["Predicted_Sentiment"] = df["text"].apply(lambda x: sentiment_pipeline(x)[0]['label'])

sentiments = []
# Iterate through the DataFrame rows safely
for i, row in df.iterrows():
    text = row['text']
    try:
        # Skip empty or non-string values
        if not isinstance(text, str) or text.strip() == "":
            sentiments.append("UNKNOWN")
            print(f"[{i}] Empty or invalid text")
            continue
        
        result = sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens (safe for most models)
        label = result['label']
        sentiments.append(label)
        print(f"[{i}] Text: {text}\n   Sentiment: {label}\n")

    except Exception as e:
        sentiments.append("ERROR")
        print(f"[{i}] Error processing text: {text}\n   Error: {e}\n")

# Add the results to your DataFrame
df["Predicted_Sentiment"] = sentiments


# Count how many are predicted as POSITIVE or NEGATIVE
sentiment_counts = df["Predicted_Sentiment"].value_counts()
print(sentiment_counts)

print("##############")
# Load the tokenizer and model from the saved directory
model_path = "saved_models/baseline_model"

model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)


#df["Sentiment_fine"] = df["text"].apply(lambda x: sentiment_pipe(x)[0]['label'])
sentiments = []
# Iterate through the DataFrame rows safely
for i, row in df.iterrows():
    text = row['text']
    try:
        # Skip empty or non-string values
        if not isinstance(text, str) or text.strip() == "":
            sentiments.append("UNKNOWN")
            print(f"[{i}] Empty or invalid text")
            continue
        
        result = sentiment_pipe(text[:512])[0]  # Truncate to 512 tokens (safe for most models)
        label = result['label']
        sentiments.append(label)
        print(f"[{i}] Text: {text}\n   Sentiment: {label}\n")

    except Exception as e:
        sentiments.append("ERROR")
        print(f"[{i}] Error processing text: {text}\n   Error: {e}\n")
        
# Add the results to your DataFrame
df["Sentiment_fine"] = sentiments

sentiment_counts2 = df["Sentiment_fine"].value_counts()
print(sentiment_counts2)    