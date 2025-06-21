import pandas as pd

# Load the data
df = pd.read_csv("final_sentiment.csv")

# Predicted NEGATIVE but actually POSITIVE
neg_pred_pos_actual = df[(df['Predicted_Sentiment'] == 'NEGATIVE') & (df['Sentiment_fine'] == 'POSITIVE')]

# Predicted POSITIVE but actually NEGATIVE
pos_pred_neg_actual = df[(df['Predicted_Sentiment'] == 'POSITIVE') & (df['Sentiment_fine'] == 'NEGATIVE')]

# Both predictions and actuals are the same
agreement = df[df['Predicted_Sentiment'] == df['Sentiment_fine']]

# Show a few examples from each
print("Predicted NEGATIVE but actual POSITIVE:")
print(neg_pred_pos_actual[['text', 'Predicted_Sentiment', 'Sentiment_fine']].head())

print("\nPredicted POSITIVE but actual NEGATIVE:")
print(pos_pred_neg_actual[['text', 'Predicted_Sentiment', 'Sentiment_fine']].head())

print("\nAgreements (Predicted == Actual):")
print(agreement[['text', 'Predicted_Sentiment', 'Sentiment_fine']].head())