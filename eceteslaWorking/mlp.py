import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk import word_tokenize     
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


stop_words = set(stopwords.words('english'))

def load_data(subset = None):
    data = pd.read_csv("../dataset_20000.csv")
    if subset is not None:
        return data.head(subset)
    return data

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def tokenize(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    # stems = stem_tokens(filtered, stemmer)
    return filtered

def sent_vectorizer(sent):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = word_model[w]
            else:
                sent_vec = np.add(sent_vec, word_model[w])
            numw+=1
        except:
            pass
    
    return np.asarray(sent_vec) / numw

def build_model(estimator):
    clf = OneVsRestClassifier(estimator=estimator)
    model = clf.fit(extract_vector_array(transformed_x_train.values), y_train)
    
    predictions = model.predict(extract_vector_array(transformed_x_test.values))
    report = classification_report(y_test, predictions)
    return report

def extract_vector_array(test):
    arr = []
    
    for i, vector in enumerate(test):
        arr.append([])
        for j, elem in enumerate(vector):
            arr[i].append(elem)
            
    return arr

word_dict = set()
def remove_stopwords(sent):
	tokens = nltk.word_tokenize(sent)
	new_sent = []

	for token in tokens:
		word = token.lower()
		if word not in stop_words and len(token) > 2:
			if word not in word_dict:
				word_dict.add(word)

			new_sent.append(word)

	return new_sent

def build_neural_net(hidden_layers, activation='relu', early_stopping=False):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, early_stopping=early_stopping, activation=activation)
    model = clf.fit(extract_vector_array(transformed_x_train.values), y_train)
    
    predictions = model.predict(extract_vector_array(transformed_x_test.values))
    report = classification_report(y_test, predictions)
    return report

def build_rnn(num_words, input_length):
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import LSTM
	from keras.layers.embeddings import Embedding
	from keras.preprocessing import sequence

	np.random.seed(7)

	embed_dim = 128
	lstm_out = 200

	model = Sequential()
	model.add(Embedding(num_words, embed_dim,input_length = input_length, dropout = 0.2))
	model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
	model.add(Dense(14, activation='softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	return model

dataset = load_data(subset = 5000)
stemmer = PorterStemmer()
x = dataset["summary"]
y = dataset.drop(["summary"], axis=1)

X = [remove_stopwords(sent) for sent in x.values]
tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, padding='post')

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

# transformed_x_test = x_test.map(sent_vectorizer)

# transformed_x_train = x_train.map(sent_vectorizer)

# report = build_neural_net(hidden_layers=(128,128), activation='relu', early_stopping=False)

#rnn
print(X.shape[1])
model = build_rnn(len(word_dict) + 1, X.shape[1])
model.fit(X_train, Y_train, batch_size=32, epochs=1,  verbose=5)

print(model.summary())


