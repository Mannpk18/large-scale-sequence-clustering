from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd

class KmerClusterer:
    def __init__(self, k=4):
        self.k = k
        self.sequences = []
        self.vectorizer = None
        self.kmer_matrix = None

    def load_sequences(self, sequences):
        self.sequences = sequences

    def _kmerize(self, sequence):
        return [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]

    def extract_kmer_features(self):
        corpus = [" ".join(self._kmerize(seq)) for seq in self.sequences]
        self.vectorizer = CountVectorizer()
        self.kmer_matrix = self.vectorizer.fit_transform(corpus)

    def cluster_kmeans(self, n_clusters=5):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(self.kmer_matrix)
        return labels
