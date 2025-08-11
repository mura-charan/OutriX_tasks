import json
import random
import numpy as np
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load intents.json
with open("intents.json") as file:
    data = json.load(file)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []

# Data preprocessing
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize & lower case
words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalnum()]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Prepare training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Convert to arrays
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Encode labels
encoder = LabelEncoder()
train_y_labels = encoder.fit_transform([doc[1] for doc in documents])

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compile and train
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save words & classes
import pickle
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
model.save("chatbot_model.h5")
