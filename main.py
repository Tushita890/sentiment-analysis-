# it is based on first csv file datasets named amazon.csv 
import numpy as np
import pandas as pd

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import re
from textblob import TextBlob
from wordcloud import WordCloud

import seaborn as sns
import matplotlib.pyplot as plt

import cufflinks as cf
%matplotlib inline

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
cf.go_offline()

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

pd.set_option("display.max_columns", None)

df=pd.read_csv(r"C:\Users\HP\Downloads\amazon.csv")
df.head()

df.reviewText.head()

review_example = df.reviewText[2031]
review_example



rt = lambda x: re.sub("[^a-zA-Z]", '', str(x))
df["reviewText"] = df["reviewText"].map(rt)
df["reviewText"] = df["reviewText"].str.lower()
df.head()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd

# Create analyzer once (faster)
analyzer = SentimentIntensityAnalyzer()

# Get polarity and subjectivity
df[['polarity', 'subjectivity']] = (
    df['reviewText']
    .apply(lambda text: pd.Series(TextBlob(str(text)).sentiment))
)

# Iterate through reviews
for index, row in df['reviewText'].items():   # changed iteritems() → items()

    score = analyzer.polarity_scores(str(row))

    neg = score["neg"]
    pos = score["pos"]

    if neg > pos:
        df.loc[index, "sentiment"] = "Negative"
    elif pos > neg:
        df.loc[index, "sentiment"] = "Positive"
    else:
        df.loc[index, "sentiment"] = "Neutral"

df[df['sentiment']=='Positive'].sort_values("wilson_lower_bound" , ascending = False).head(5)


def categorical_variable_summary(df, column):
    print(df[column].value_counts())
    print(df[column].value_counts(normalize=True) * 100)

# Call function
categorical_variable_summary(df, 'sentiment')

import matplotlib.pyplot as plt

# Get counts of sentiment values
sentiment_counts = df['sentiment'].value_counts()

# 1. Bar chart
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar')
plt.title("Sentiment Distribution - Bar Chart")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


# 2. Pie chart
plt.figure(figsize=(6,6))
sentiment_counts.plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.title("Sentiment Distribution - Pie Chart")
plt.ylabel("")
plt.show()


# 3. Line chart
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='line', marker='o')
plt.title("Sentiment Distribution - Line Chart")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
df.head()

