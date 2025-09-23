import numpy as np

import re
import time
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import twitter_samples, stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc
from helper import get_llm, eval_prompt

np.random.seed(42)

llm_lists = [
    # "gemini-2.0-flash-lite",
    # "gemini-2.5-pro",
    "gemini-2.5-flash",
    # "gpt-4-turbo",
    # "gpt-4",
]

# nltk.download('twitter_samples')
# nltk.download('stopwords')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = np.random.choice(all_positive_tweets[4000:], 20, replace=False).tolist()
test_neg = np.random.choice(all_negative_tweets[4000:], 20, replace=False).tolist()

test_x = test_pos + test_neg
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

def run_benchmark():
    for model in llm_lists:
        llm = get_llm(model_name=model)
        y_pred = np.zeros(test_y.shape)
        for i, tweet in enumerate(test_x):
            response = llm.invoke(eval_prompt.format(text=tweet))
            if response:
                sentiment = re.search(r'"(positive|negative)"', response.content.lower())
                if sentiment:
                    y_pred[i] = 1 if sentiment.group(1) == "positive" else 0
            time.sleep(1)  # To avoid rate limiting

        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred)
        recall = recall_score(test_y, y_pred)
        f1 = f1_score(test_y, y_pred)
        tn, fp, fn, tp = confusion_matrix(test_y, y_pred).ravel
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        roc_auc = auc([0, fpr, 1], [0, tpr, 1])

        print(f"Model: {model}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print("-" * 30)
        

if __name__ == "__main__":
    run_benchmark()