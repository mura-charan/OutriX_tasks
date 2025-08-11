import random
import json
import pickle
import numpy as np
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (only once needed)
nltk.download('punkt')
nltk.download('wordnet')

# Load trained model and files
lemmatizer = WordNetLemmatizer()
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")
intents = json.load(open("intents.json"))

# Preprocess user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.isalnum()]
    return sentence_words

# Convert sentence to bag-of-words vector
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Generate response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't get that. Can you rephrase?"

    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Start chatting
print("🤖 ChatBot is ready to talk! (Type 'quit' to exit)\n")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: Bye! Have a great day 🤗")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)
