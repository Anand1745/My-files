import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay

df = pd.read_csv('C:\Users\anand\OneDrive\Desktop\Projects\image_classification\vaccination_tweets')

df.head()

df.info()

df.isnull().sum()

df.columns

text_df = df.drop(['id', 'user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date', 'hashtags', 'source', 'retweets', 'favorites',
       'is_retweet'], axis=1)
text_df.head()

print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")

text_df.info()

def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

text_df.text = text_df['text'].apply(data_processing)

text_df = text_df.drop_duplicates('text')

stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

text_df['text'] = text_df['text'].apply(lambda x: stemming(x))

text_df.info()

def polarity(text):
    return TextBlob(text).sentiment.polarity

text_df['polarity'] = text_df['text'].apply(polarity)

text_df.head(10)

def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

text_df['sentiment'] = text_df['polarity'].apply(sentiment)

text_df.head()

fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = text_df)

fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen","gold","red")
wp = ('linewidth':2, 'edgecolor':"black")
tags = text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%',shadow=True,colors=colors,startangle=90,wedgeprops=wp,explode=explode,label='')
plt.title('Distribution of sentiments')

pos_tweets = text_df[text_df.sentiment =='Positive']
pos_tweets = pos_tweets.sort_values(['polarity'],ascending=False)
pos_tweets.head()

text = ' '.join([word for word in pos_tweet['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud =WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show()

neg_tweets = text_df[text_df.sentiment =='Negetive']
neg_tweets = neg_tweets.sort_values(['polarity'],ascending=False)
neg_tweets.head()

text = ' '.join([word for word in neg_tweet['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud =WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negetive tweets', fontsize=19)
plt.show()

neutral_tweets = text_df[text_df.sentiment =='Neutral']
neutral_tweets = neg_tweets.sort_values(['polarity'],ascending=False)
neutral_tweets.head()

text = ' '.join([word for word in neutral_tweet['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud =WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()

vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['text'])

feature_names +vect.get_feature_names_out()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features:\n {}".format(feature_names[:20]))

X = text_df['text']
Y = text_df['sentiment']
X = vect.transform(X)

x_train, x_text, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))

import warnings
warnings.filtewarnings('ignore')

