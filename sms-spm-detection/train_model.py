import pandas as pd
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(ps.stem(word))

    return " ".join(y)

# 1. Load your dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# 2. Encode labels (ham -> 0, spam -> 1)
le = LabelEncoder()
df['label_num'] = le.fit_transform(df['label'])  # spam = 1, ham = 0

# 3. Preprocess text
df['transformed_text'] = df['text'].apply(transform_text)

# 4. TF-IDF Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['transformed_text'])
y = df['label_num']

# 5. Train the model
model = MultinomialNB()
model.fit(X, y)

# 6. Save the model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model and vectorizer saved successfully!")
