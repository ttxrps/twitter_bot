from tabnanny import check
import tweepy
import time
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd
import pickle
import numpy as np
import random

from xgboost import XGBClassifier

API_key= "Your API Key"
API_secret_key="Your API secret key"
access_token="Your Access token"
access_token_secret="Your Access token secret"

loaded_model = pickle.load(open('Model_name.pkl', 'rb')) # use your model
positive_reply = pickle.load(open('Reply/list_reply_positive.data', 'rb'))
negative_reply = pickle.load(open('Reply/list_reply_negative.data', 'rb'))
ambiguous_reply = pickle.load(open('Reply/list_reply_ambiguous.data', 'rb'))
google_model = pickle.load(open('w2vModel_name.pkl', 'rb')) # use your  w2v model
# list_word = pickle.load(open('words.data', 'rb'))
# get_vector = pickle.load(open('list_vector.data', 'rb'))


auth = tweepy.OAuthHandler("API_key",
                            "API_secret_key")
auth.set_access_token("access_token",
                        "access_token_secret")
api = tweepy.API(auth, wait_on_rate_limit=True)


def text_processing(text):
    text = str(text)

    # remove single quotes
    text = text.replace("'", " ")
    text = re.sub("[NAME]", " ", text)
    text = text.lower()
    text = re.sub("(@[\w]+)", " ", text)
    text = re.sub("(#[\w]+)", " ", text)
    text = re.sub('[\W]+', " ", text)
    text = re.sub("\d+", " ", text)
    # word tokenization using text-to-word-sequence
    text = re.sub(
        "[\.\,\!\?\:\;\-\=\^\(\)\_\%\&\*\$\+\/\—\`\<\>\[\]\'\’\“\”]+", " ", text)
    text = re.sub("tl dr", " ", text)
    stop_words = set(stopwords.words('english'))
    text = " ".join(i for i in text.split() if not i in stop_words)
    text = " ".join(i for i in text.split() if not len(i) < 2)

    # Stemming
    stemmer = PorterStemmer()

    stem_input = nltk.word_tokenize(text)
    stem_text = ' '.join([stemmer.stem(word) for word in stem_input])

    # lemmatization
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        #  """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    lem_input = nltk.word_tokenize(stem_text)
    lem_text = ' '.join([lemmatizer.lemmatize(
        w, get_wordnet_pos(w)) for w in lem_input])

    return lem_text


def random_vector(text):
    vector = np.zeros(300,)
    for i in range(999):
        word = random.choice(text.split())
        try:
            word_vec = google_model[word]
            vector += word_vec
        except:
            pass
    return vector/1000


def vectors(text):
    global word_embeddings
    word_embeddings = []

    avgword2vec = None
    count = 0
    avg_vec = random_vector(text)
    for word in text.split():
        count += 1
        if word in google_model.wv.vocab:
            print('word in list')
            if avgword2vec is None:
                avgword2vec = google_model[word]
            else:
                avgword2vec = avgword2vec + google_model[word]
        else:
            if avgword2vec is None:
                avgword2vec = avg_vec
            else:
                avgword2vec = avgword2vec + avg_vec
        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
        word_embeddings.append(avgword2vec)
        return word_embeddings

def get_text_reply(result):
    text_reply = ""
    if result == 0:
        text_reply = random.choice(ambiguous_reply)
    elif result == 1:
        text_reply = random.choice(positive_reply)
    elif result == -1:
        text_reply = random.choice(negative_reply)
    return text_reply



def mention_checker():
    print("mention")
    file_name = "twit.txt"
    get_new_mentions = True
    while get_new_mentions:
        try:
            current_mentions = []          
            toReply = ""
            with open(file_name) as f:
                for line in f:
                    line = line.strip("\n")
                    current_mentions.append(line)
            with open(file_name, "a") as f:
                for mention in api.mentions_timeline(count=1):
                    f.write(mention.text + "\n")
                    toReply = mention.user.screen_name
            new_mentions = []
            with open(file_name) as f:
                for line in f:
                    line = line.strip("\n")
                    new_mentions.append(line)
            current_mentions_set = set(current_mentions)
            new_mentions_set = set(new_mentions)
            new_mentions_list = new_mentions_set.difference(
                current_mentions_set)
            tweets = api.user_timeline(screen_name=toReply, count=1)
            for new_mention in new_mentions_list:
                print(new_mention)
                text_predict = text_processing(new_mention)
                print(text_predict)
                df_text = vectors(text_predict)
                print(df_text)
                result = loaded_model.predict(df_text)
                print(result[0])
                text_reply = get_text_reply(result[0])
                print(text_reply)
                for tweet in tweets:
                    api.update_status(
                        "@" + toReply + " " + str(text_reply), in_reply_to_status_id=tweet.id)
                    print("reply success")
            time.sleep(15)
        except tweepy.errors.TwitterServerError:
            print('You have exceeded the API rate limit.')


mention_checker()
