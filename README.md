# Sentiment Analysis Using Python

This Python code performs sentiment analysis using a rule-based method. It analyzes the sentiment of the text based on the presence of positive and negative words.

## Imports

```python
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')

positive_words = ['good', 'together' , 'pdf' , 'excellent', 'awesome', 'happy', 'enjoying', 'excited', 'collaborative', 'productive', 'good luck', 'safe', 'joined', ':+1:', ':heart:', ':relaxed:', ':tada:']
negative_words = ['bad', 'terrible', 'horrible', 'awful', 'sad', 'disappointing', 'dreadful', 'annoying', 'frustrating', 'unpleasant', 'frustrating']

```

In this section, we import the required libraries: pandas for data manipulation, nltk (Natural Language Toolkit) for natural language processing, and specifically, word_tokenize from nltk.tokenize to tokenize the text into words. We also define two lists, positive_words and negative_words, which contain words associated with positive and negative sentiments, respectively.

## Sentiment Analysis Function

```python
def calculate_sentiment_score(text):
    if pd.notnull(text):
        tokens = word_tokenize(text.lower())
        sentiment_score = 0
        for token in tokens:
            token = token.strip(string.punctuation)
            if token in positive_words:
                sentiment_score += 1
            elif token in negative_words:
                sentiment_score -= 1
        return sentiment_score
    return 0

def map_to_0_to_100(score):
    max_score = 5
    min_score = -5
    scaled_score = ((score - min_score) / (max_score - min_score)) * 100
    return min(max(scaled_score, 0), 100)

def calculate_name_sentiment_score(name_group):
    sentiment_scores = name_group['Text'].apply(calculate_sentiment_score)
    message_lengths = name_group['Text'].str.len()
    
    if message_lengths.sum() > 0:
        weighted_score = (sentiment_scores * message_lengths).sum() / message_lengths.sum()
        return map_to_0_to_100(weighted_score)
    else:
        return 0
```

The sentiment_analysis function takes a text input as a parameter and performs sentiment analysis on that text. It first tokenizes the input text into individual words using word_tokenize from NLTK. Then, it calculates a sentiment score by counting the occurrences of positive and negative words. If a word is found in the positive_words list, the sentiment score is increased by 1, and if a word is found in the negative_words list, the sentiment score is decreased by 1. The function returns the sentiment label as 'Positive', 'Negative', or 'Neutral' based on the sentiment score.

## Performing Sentiment Analysis on Dataset

```python
# Assuming you have the dataset in a CSV file named 'dataset.csv'
df = pd.read_csv(r'C:\Users\akagr\OneDrive\Desktop\XtraLeap\Sentimental Analysis\dataset.csv')
df['Sentiment_Score'] = df['Text'].apply(calculate_sentiment_score)
name_scores = df.groupby('Name').apply(calculate_name_sentiment_score).reset_index()
name_scores.columns = ['Name', 'Name_Sentiment_Score']
name_scores = name_scores[name_scores['Name_Sentiment_Score'] > 0]

print(name_scores)

```

This section assumes that you have the dataset in a CSV file named 'file.csv'. You should replace 'path/to/your/file.csv' with the actual file path where your dataset is located. The code reads the CSV file into a pandas DataFrame (df). Then, it applies the sentiment_analysis function to each text in the 'Text' column of the DataFrame and stores the sentiment analysis results in a new column named 'Sentiment'. Finally, the DataFrame with the added 'Sentiment' column is printed to display the results.

## How It Works

1. The code imports the required libraries: pandas, nltk, and nltk.tokenize.word_tokenize.
2. Two lists, `positive_words` and `negative_words`, are defined, which contain words associated with positive and negative sentiments, respectively.
3. The `sentiment_analysis` function is defined to perform sentiment analysis on the input text:
   - It tokenizes the text into individual words.
   - It calculates a sentiment score based on the occurrence of positive and negative words in the text.
   - The function returns the sentiment label as 'Positive', 'Negative', or 'Neutral' based on the sentiment score.
4. The code reads the dataset from a CSV file named 'file.csv' and stores it in a pandas DataFrame `df`.
5. The `sentiment_analysis` function is applied to each text in the 'Text' column of the DataFrame, and the sentiment analysis results are stored in a new column named 'Sentiment'.
6. Finally, the DataFrame with the added 'Sentiment' column is printed to display the results.

## How to Use

1. Make sure you have Python, pandas, and nltk installed.
2. Install the required libraries using pip:

```bash
pip install pandas nltk
```
## Example
Suppose you have a CSV file named data.csv with the following content:

```vbnet
Name	Text
karthik	Sure sir we will tell the rest of the members to join, if everyone will be free at some time tomorrow we will take a group pic and send you.Sentimental Analysis.pdfwe are on it sirSentimental Analysis (1).pdf
keerthi	
sushma	Sure sir we will work on it together.Good evening, Siva. pdf . I hope this message finds you well. I would like to provide you with the results of my research on rule-based models for sentiment analysis. Here are the results
nikhil	Okay that would be great, thank you.
pranav	
sameer	Siva I am interested in learning, but I don't know anyone learning this .pdf
jahnavi	
mehak	Sure sir we will take group pic tomorrow and we are so excited to do this internship.No sir i don't know who is doing or learning it
jayasai	

```
After running the code, the DataFrame will be updated as follows:
```mathematica
      Name  Name_Sentiment_Score
2  karthik                  60.0
4    mehak                  60.0
5   nikhil                  50.0
7   sameer                  60.0
8   sushma                  60.0
```
