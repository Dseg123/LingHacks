import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import string
from nltk.stem import WordNetLemmatizer
  
lemmatizer = WordNetLemmatizer()

nltk.download("stopwords")
stop = list(stopwords.words("english"))
for i in range(len(stop)):
    stop[i] = ''.join([c for c in stop[i] if c not in string.punctuation])
print(stop)

model = KeyedVectors.load("word2vec.wikivectors")
print(model.most_similar("princeton"))
def label_score(tokens, label):
    tot = 0
    for token in tokens:
        if token in model:
            print(token, label, model.similarity(token, label))
            tot += model.similarity(token, label)
    return tot/len(tokens)

def get_labels(post):
    MY_LABELS = ["housing", "social", "academic", "health", "logistic", "technology"]
    post = ''.join([c for c in post if c not in string.punctuation])
    post = post.lower()
    tokens = word_tokenize(post)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop]
    print(tokens)
    scores = [[label, 0] for label in MY_LABELS]
    for i in range(len(MY_LABELS)):
        scores[i][1] = label_score(tokens, scores[i][0])
    scores.sort(key = lambda x: x[1], reverse = True)
    return scores
    
    
while True:
    s = input()
    if s == "quit":
        break
    else:
        print(get_labels(s))