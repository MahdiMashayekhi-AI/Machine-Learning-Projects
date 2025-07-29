import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    A custom transformer for basic text cleaning:
    - Lowercasing
    - Removing punctuation
    - Removing numbers
    - Removing extra whitespaces
    - Removing URLs and emails
    - Removing special characters (e.g., @, #)
    """

    def __init__(self, remove_numbers=True, remove_urls=True):
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._clean_text)

    def _clean_text(self, text):
        # Ensure the input is treated as a string
        text = str(text)

        # 1. Lowercase
        text = text.lower()

        # 2. Remove URLs and emails
        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|https\S+",
                          '', text, flags=re.MULTILINE)
            text = re.sub(r'\S+@\S+', '', text)

        # 3. Remove punctuation and special characters
        text = re.sub(r"[^\w\s]", " ", text)

        # 4. Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # 5. Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


nltk.download('stopwords', quiet=True)


class StopwordsRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove stopwords from text.
    Input: Series of strings
    Output: Series of strings with stopwords removed
    """

    def __init__(self, lang='english'):
        self.lang = lang
        self.stopwords = set(stopwords.words(lang))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._remove_stopwords)

    def _remove_stopwords(self, text):
        words = text.split()
        filtered = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered)


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


class Stemmer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to perform stemming on text.
    Input: Series of strings
    Output: Series of strings with stemmed words
    """

    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._stem_text)

    def _stem_text(self, text):
        words = word_tokenize(text)
        stemmed = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cleaner = TextCleaner()
        self.stop = StopwordsRemover()
        self.stem = Stemmer()
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm='l2'
        )

    def fit(self, X, y=None):
        # Fit each component individually
        X_cleaned = self.cleaner.fit_transform(X)
        X_stopwords_removed = self.stop.fit_transform(X_cleaned)
        X_stemmed = self.stem.fit_transform(X_stopwords_removed)
        self.vectorizer.fit(X_stemmed)
        return self

    def transform(self, X):
        # Transform each component individually
        X_cleaned = self.cleaner.transform(X)
        X_stopwords_removed = self.stop.transform(X_cleaned)
        X_stemmed = self.stem.transform(X_stopwords_removed)
        return self.vectorizer.transform(X_stemmed)
