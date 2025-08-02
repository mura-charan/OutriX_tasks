import pandas as pd

# part-1(Load the dataset)
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
data = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Show first 5 messages
print(data.head())

# Check spam vs ham count
print("\nClass distribution:")
print(data['label'].value_counts())
#part-2(cleaning)
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and apply stemming
    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(cleaned)

# Apply preprocessing to the message column
data['cleaned_message'] = data['message'].apply(preprocess_text)

# Show before and after
print(data[['message', 'cleaned_message']].head())

#part-3(feature extraction)
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['cleaned_message']).toarray()

# Convert labels to 0 (ham) and 1 (spam)
y = data['label'].map({'ham': 0, 'spam': 1}).values

# Show shape of final data
print("Feature matrix shape:", X.shape)

#part-4(training model)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

#part-5(testing)
# 📩 Predict on custom messages
while True:
    msg = input("\nEnter a message to test (or type 'exit' to quit): ")
    if msg.lower() == 'exit':
        break
    # Preprocess the input
    cleaned = preprocess_text(msg)
    # Convert to vector
    vector = tfidf.transform([cleaned]).toarray()
    # Predict
    prediction = model.predict(vector)[0]
    label = "SPAM ❌" if prediction == 1 else "HAM ✅"
    print("Prediction:", label)

