import nltk
from nltk.tokenize import word_tokenize
import pickle
import string


f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

### decompose post into features loosely based on word order ###
def dialogue_act_features(post):
    features = {}
    post = post.lower()
    post = ''.join([c for c in post if c not in string.punctuation])
    i = 0
    for word in nltk.word_tokenize(post):
        features['words' + str(i)] = word
        i += 1
    return features

### return probdist given by Naive Bayes Classifier for post being a question ###
def is_question_1(post):
    c = (classifier.prob_classify(dialogue_act_features(s)))
    return [c.prob(True), c.prob(False)]

### use hard-coded interrogatives to check if it is an obvious question ###
def is_question_2(post):
    post = post.lower()
    post = ''.join([c for c in post if c not in string.punctuation])
    post = " " + post + " "
    
    words1 = ["do", "does", "am", "is", "isnt", "are", "arent", "was", "wasnt", "were", "werent", "will", "wont", "should", "shouldnt", "could", "couldnt", "would", "wouldnt"]
    words2 = ["who", "what", "when", "where", "why"]
    words3 = ["it", "they", "i", "we", "that", "those", "this"]
    combos = []
    for x in words2:
        for y in words1:
            combos.append(" " + x + " " + y + " ")
    for x in words1:
        for y in words3:
            combos.append(" " + x + " " + y + " ")
    for i in combos:
        if i in post:
            print(i)
            return True
    return False

### Identify if question through a combination of classifier and hard-coded methods ###
def is_question(post):
    if "?" not in post:
        return False
    postNew = ''.join([c for c in post if c not in string.punctuation])
    postNew = postNew.lower()
    pl = nltk.word_tokenize(postNew)
    print(is_question_1(post))
    truthVec = ["?" in post, is_question_1(post)[0] > 0.9, is_question_2(post)]

    if len(pl) >= 5 and (truthVec[1] or truthVec[2]):
        return True
    return False


### Testing code -- keep entering questions to see if model works ###
while True:
    s = input()
    if s == "quit":
        break
    else:
        print("the statement: '", s, "' is classified as: ", is_question(s))


### Given a post, first check if it is at least 5 words and contains question mark ###
### then run it through interrogatives to weed out obvious questions ###
### finally if classifier passes a high threshold then probably a question ###