import nltk
from nltk import RegexpTokenizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import matplotlib.pyplot as plt
#import PyPDF2
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import pyLDAvis.sklearn
from plotly.offline import init_notebook_mode, iplot
import pyLDAvis.sklearn
#import spacy
from math import log
import networkx as nx
from pyvis.network import Network
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer




def removeStopWords(lText, lStopWords, wordlen):
    """
    Removes and lowers the case of the words not in the stop words
    :param lText: List of words that you want to remove words from
    :param lStopWords: List of words that need to be removed
    :return: List - final list of words without the stopwords
    """
    lResult = []
    vocab = []
    for word in lText:
        if word not in lStopWords and len(word) > wordlen:
            lResult.append(word.lower())
            if word.lower() not in vocab:
                vocab.append(word.lower())
    return lResult,vocab


# Reference Sample Python provided in Course 626-A
def ngram(text, grams):
    n_grams_list = []
    count = 0
    for token in text[:len(text) - grams + 1]:
        n_grams_list.append(text[count]+' '+text[count+grams-1])
        count=count+1

    return n_grams_list

# Reference Sample Python provided in Course 626-A
def chunk_replacement(chunk_list, text):
    """    Connects words chunks in a text by joining them with an underscore.
    :param chunk_list: word chunks
    :type chunk_list: list of strings/ngrams
    :param text:
    text    :type
    text: string
    :return: text with underscored chunks
    :type: string    """

    for chunk in chunk_list:
        text = text.replace(chunk, chunk.replace(' ', '_'))
        return text

# Reference Sample Python provided in Course 626-A
def most_common(lst, num, lStop):
    data = Counter(lst)
    common = data.most_common(num)
    top_comm = []
    for i in range (0, num):
        if not any(map(common[i][0].__contains__, lStop)):
                top_comm.append (common[i][0])
    return top_comm


# Reference Sample Python provided in Course 626-A
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
               for i in topic.argsort()[:-top_n - 1:-1]])

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    #https: // www.analyticsvidhya.com / blog / 2018 / 10 / mining - online - reviews - topic - modeling - lda /
    return tag_dict.get(tag,wordnet.NOUN)



def lemmatize_stemming(lText):
    lStemmed  = []
    for word in lText:
        tagged_word = get_wordnet_pos(word)
        if tagged_word in ['n']:
            lStemmed.append(WordNetLemmatizer().lemmatize(word,tagged_word))
    return lStemmed



tokenizer = RegexpTokenizer(r'\w+')

# Load the stop words
hStopWords = open("stopwords_en.txt","r")
tStopWords = hStopWords.read()
lStopWords = tokenizer.tokenize(tStopWords)
additionalStopWords  = ["looked","said","didn","know","couldn","felt","really","just","like","couldn","does","this","phone"]
lStopWords = lStopWords + additionalStopWords

dtPhoneList = pd.read_csv("20191226-items.csv",sep=',')
dtReviews = pd.read_csv("20191226-reviews.csv",sep =',')

dtReviews = dtReviews.loc[dtReviews.rating.isin([1,5])]

strReviewTxt = dtReviews['body'].str.cat(sep = ' ')

tokenizer = RegexpTokenizer(r'\w+')
print("\nTokenizing the words..")
lRawText = tokenizer.tokenize(strReviewTxt)

print('Removing the stop words..')
lCleanedReviews, vocabulary = removeStopWords(lRawText,lStopWords,3)

print("Identifying the bigrams")
# Reference Sample Python provided in Course 626-A
all_bigrams_list = ngram(lCleanedReviews, 2)
# all_bigrams_list.append(ngram(lCleanedReviews, 3))

lStopBiGrams = ['good','great']
top_bigrams = most_common(all_bigrams_list,30,lStopBiGrams)

full_string = ' '.join(lCleanedReviews)

text_chunked = chunk_replacement(top_bigrams, full_string)
text_chunked_lst = list(text_chunked.split(' '))

print("Lemmatize the data")
text_chunked_lst_lemm = lemmatize_stemming(text_chunked_lst)

print("\nPrinting the Word Cloud")
# Defining the wordcloud parameters
wc = WordCloud(background_color="white", max_words=2000,
               stopwords=lStopWords)

wc.generate(' '.join(map(str, text_chunked_lst_lemm)))
# Store to file
wc.to_file('Rating 1 Word Cloud.png')
plt.imshow(wc)
plt.axis('off')
plt.show()


# Using LDA to extract the topics from the reviews
vectorizer = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(text_chunked_lst_lemm)

lda = LatentDirichletAllocation(n_components=5, max_iter=10, learning_method='online',verbose=False)
data_lda = lda.fit_transform(data_vectorized)

# Keywords for topics clustered by Latent Dirichlet Allocation
print('\nLDA Model:')
selected_topics(lda, vectorizer)

dash = pyLDAvis.sklearn.prepare(lda, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(dash, 'LDA_Visualization.html')