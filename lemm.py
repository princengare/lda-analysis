import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = ["closing.txt", "question1.txt", "question2.txt", "question3.txt", "question4.txt", "question5.txt", "question6.txt", "question7.txt", "question8.txt", "question9.txt", "question10.txt", "question11.txt", "question12.txt"]

# Initialize accumulators for all files
all_stemmed_words = []
all_lemmatized_words = []

# Create stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define stopwords and punctuation set
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Process each file
for j in text:
    file = open(j, "r")
    data = file.read()
    print(f"Processing {j}...")

    # Tokenize into sentences
    sentences = nltk.sent_tokenize(data)

    # Stemming process
    for sentence in sentences:
        # Tokenize into words
        words = nltk.word_tokenize(sentence)
        # Filter out stopwords and punctuation, and apply stemming
        stemmed_words = [stemmer.stem(word) for word in words if word.lower() not in stop_words and word not in punctuation]
        all_stemmed_words.extend(stemmed_words)

    # Lemmatization process
    for sentence in sentences:
        # Tokenize into words
        words = nltk.word_tokenize(sentence)
        # Filter out stopwords and punctuation, and apply lemmatization
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words and word not in punctuation]
        all_lemmatized_words.extend(lemmatized_words)

# Frequency distributions for combined words
stemmed_freq_dist = FreqDist(all_stemmed_words)
lemmatized_freq_dist = FreqDist(all_lemmatized_words)

# Print the 50 most common words after stemming (for all files combined)
print('Top 50 words after stemming:')
for word, frequency in stemmed_freq_dist.most_common(50):
    print(f'{word}: {frequency}')

print("\n" + "-"*50 + "\n")

# Print the 50 most common words after lemmatization (for all files combined)
print('Top 50 words after lemmatization:')
for word, frequency in lemmatized_freq_dist.most_common(50):
    print(f'{word}: {frequency}')
