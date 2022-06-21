from sense2vec import Sense2Vec

import re, string
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('universal_tagset')

# Predefined topics; usage determined for Princeton Class of 2026 Discord Server
# Key, value pair for official name and spaCy compatible usage, respectively
TOPICS = {
    "Academics": "academics|NOUN",
    "Student Life": "student_life|NOUN",
    "Dorms": "dorms|NOUN",
    "Food": "food|NOUN",
    "Placement": "placement|NOUN",
    "Technology": "technology|NOUN",
    "Science": "science|NOUN",
    "Humanities": "humanities|NOUN",
}

# https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e

# Tag everything (POS)
# Preprocess/clean up (maintaining tags)
# Remove stopwords
# Lemmatize (base form of words)
# Iterate over words and average their similarity to topic headings

# initialize the lemmatizer
wl = WordNetLemmatizer()

# initialize sense2vec
s2v = Sense2Vec().from_disk("/Users/carter/Downloads/s2v_old")


def add_pos_tags(s):
    '''
    Given a string, add spaCy compatible POS tags for each tokenized word

    Support for single sentences and series
    '''

    # Universal expands the tag abbreviations (making it compatible with spaCy)
    return nltk.pos_tag(word_tokenize(s), tagset='universal')


# print(add_pos_tags('Anyone who decorated their cap for graduation, what did you use to stick stuff to the cap?'))

# convert to lowercase, strip and remove punctuations
def pre_process(t):
    '''
    Given a list of tuples (word, POS), return a preprocessed list of tuples

    Convert to lowercase, strip whitespace, remove punctuation
    '''
    new = []
    for pair in t:
        # remove punctuation (. is the POS tag for the universal tagset)
        if pair[1] == '.': continue

        word = pair[0].lower().strip()
        word = re.compile('<.*?>').sub('', word) 
        word = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', word)  
        word = re.sub('\s+', ' ', word)  
        word = re.sub(r'\[[0-9]*\]',' ',word) 
        word = re.sub(r'[^\w\s]', '', str(word).lower().strip())
        word = re.sub(r'\d',' ',word) 
        word = re.sub(r'\s+',' ',word)

        new.append((word, pair[1]))
    
    return new

# print(pre_process(add_pos_tags('Anyone who decorated their cap for graduation, what did you use to stick stuff to the cap?')))

# STOPWORD REMOVAL
def remove_stopwords(t):
    '''
    Given a list of tuples (likely from the pre-process pipeline stage) of format (word, POS), remove
    stopwords from the english lexicon

    Returns another list of tuples
    '''
    # prechecks if the iterable is a tuple and of length two (word, POS) and then removes stopwords
    res = filter(lambda x: (type(x) == tuple and len(x) == 2) and x[0] not in stopwords.words('english'), t)

    return list(res)

# print(remove_stopwords(pre_process(add_pos_tags('Anyone who decorated their cap for graduation, what did you use to stick stuff to the cap?'))))

 
def get_wordnet_pos(tag):
    '''
    Helper function that takes expanded POS (spaCy compatible)
    and returns compatible POS tags for the wordnet lemmatizer
    '''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatizer(t):
    '''
    Given a list of tuples (from the pipeline) in format (word, POS), lemmatize all words
    
    Return a new list of tuples
    '''
    new = []
    for pair in t:
        new.append((
            wl.lemmatize(pair[0], get_wordnet_pos(pair[1])),  # lemmatized word
            pair[1]                                           # persist POS tag (spaCy compatible)
        ))

    return new


# print(lemmatizer(remove_stopwords(pre_process(add_pos_tags('Anyone who decorated their cap for graduation, what did you use to stick stuff to the cap?')))))


def post_process(string):
    '''
    Runs pipeline methods which returns list of tuples (word, POS)

    Returns list with remaining words and POS in form word|POS (spaCy compatible)
    '''

    res = lemmatizer(remove_stopwords(pre_process(add_pos_tags(string))))

    final = []
    for t in res:
        final.append(t[0] + '|' + t[1])

    return final


# print(post_process('Anyone who decorated their cap for graduation, what did you use to stick stuff to the cap?'))

def determine_similarity(l, topic):
    '''
    Given a list of word|POS pairs, average the similarity to a given topic using sense2vec's similarity method

    Return a float 0 to 1, inclusive
    '''

    total = 0

    for w in l:
        total += s2v.similarity(w, topic)
        # print(w, topic, s2v.similarity(w, topic))

    return total / len(l)

def main(s, n):
    '''
    Given a string, runs it through the pipeline and determines top n best matching labels
    '''

    res = post_process(s)

    label_avgs = {}

    for topic, spacy_topic in TOPICS.items():
        try:
            avg_sim = determine_similarity(res, spacy_topic)
            label_avgs[topic] = avg_sim
        except:
            print('An error occured when running sense2vec\'s similarity method')
    
    # stores top n results based on float value in dictionary
    # default to two if overflow
    top_labels = sorted(label_avgs.items(), key=lambda x:-x[1])[:(n if n < len(label_avgs) - 1 else 2)]

    label_list = []
    for label in top_labels:
        label_list.append(label[0])

    return label_list 


number_of_labels = 3
print(main('Anyone who decorated their cap for graduation, what did you use to stick stuff to the cap?', number_of_labels))

# # initializes argument parser
# parser = argparse.ArgumentParser(description='Given a question, determine an appropriate label')

# parser.add_argument("question", metavar='q', type=string, help = "Question to be given a label")
# parser.add_argument("-n", dest='num', help = "Number of labels to return, defaults to 2")


# args = parser.parse_args()

# if args.question:
#     if args.num:
#         main(args.question, args.num)
#     else:
#         main(args.question, 2)
