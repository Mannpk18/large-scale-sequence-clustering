#!/usr/bin/env python3
"""
Neural Network-Based DNA Sequence Clustering
Implements autoencoder and CNN-based approaches for DNA clustering
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class NeuralDNAClusterer:
    def __init__(self, max_length=None):
        self.sequences = []
        self.max_length = max_length
        self.encoded_seqs = None
        self.embeddings = None
        self.labels = None
        self.autoencoder = None
        self.encoder = None
        
    def load_sequences(self, sequences):
        """Load DNA sequences"""
        self.sequences = [seq.upper() for seq in sequences]
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in self.sequences)
        print(f"Loaded {len(self.sequences)} sequences, max length: {self.max_length}")
        return self
    
    def encode_sequences(self):
        """One-hot encode DNA sequences"""
        print("One-hot encoding sequences...")
        
        # Create mapping
        base_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        
        encoded = []
        for seq in self.sequences:
            # Pad or truncate sequence
            if len(seq) < self.max_length:
                seq = seq + 'N' * (self.max_length - len(seq))
            else:
                seq = seq[:self.max_length]
            
            # One-hot encode
            seq_encoded = np.zeros((self.max_length, 5))
            for i, base in enumerate(seq):
                if base in base_to_int:
                    seq_encoded[i, base_to_int[base]] = 1
                else:
                    seq_encoded[i, 4] = 1  # Unknown base
            
            encoded.append(seq_encoded)
        
        self.encoded_seqs = np.array(encoded)
        print(f"Encoded sequences shape: {self.encoded_seqs.shape}")
        return self.encoded_seqs
    
    def build_autoencoder(self, encoding_dim=32):
        """Build autoencoder for dimensionality reduction"""
        if self.encoded_seqs is None:
            self.encode_sequences()
        
        input_dim = self.max_length * 5  # Flattened one-hot encoding
        
        # Encoder
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Models
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        print(f"Built autoencoder with encoding dimension: {encoding_dim}")
        return self.autoencoder
    
    def build_cnn_autoencoder(self, encoding_dim=32):
        """Build CNN-based autoencoder for sequence patterns"""
        if self.encoded_seqs is None:
            self.encode_sequences()
        
        # Encoder
        input_layer = keras.Input(shape=(self.max_length, 5))
        
        # Convolutional layers
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
        
        # Flatten and encode
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(x)
        
        # Decoder
        x = layers.Dense(64, activation='relu')(encoded)
        x = layers.Dense(8 * (self.max_length // 4), activation='relu')(x)
        x = layers.Reshape((self.max_length // 4, 8))(x)
        
        # Deconvolutional layers
        x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        decoded = layers.Conv1D(5, 3, activation='sigmoid', padding='same')(x)
        
        # Models
        self.autoencoder = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)
        
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        print(f"Built CNN autoencoder with encoding dimension: {encoding_dim}")
        return self.autoencoder
    
    def train_autoencoder(self, epochs=50, batch_size=32, validation_split=0.2):
        """Train the autoencoder"""
        if self.autoencoder is None:
            self.build_autoencoder()
        
        # Prepare data
        if self.autoencoder.input_shape[-1] == self.max_length * 5:
            # Dense autoencoder - flatten sequences
            X = self.encoded_seqs.reshape(len(self.encoded_seqs), -1)
        else:
            # CNN autoencoder - keep 3D shape
            X = self.encoded_seqs
        
        print("Training autoencoder...")
        history = self.autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=True
        )
        
        # Plot training history
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # Show reconstruction example
        if len(X) > 0:
            original = X[0:1]
            reconstructed = self.autoencoder.predict(original, verbose=0)
            
            if self.autoencoder.input_shape[-1] == self.max_length * 5:
                original = original.reshape(self.max_length, 5)
                reconstructed = reconstructed.reshape(self.max_length, 5)
            else:
                original = original[0]
                reconstructed = reconstructed[0]
            
            plt.plot(original.flatten()[:50], label='Original', alpha=0.7)
            plt.plot(reconstructed.flatten()[:50], label='Reconstructed', alpha=0.7)
            plt.title('Reconstruction Example (first 50 values)')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return history
    
    def extract_embeddings(self):
        """Extract embeddings using trained encoder"""
        if self.encoder is None:
            print("No trained encoder available")
            return None
        
        # Prepare data
        if self.encoder.input_shape[-1] == self.max_length * 5:
            X = self.encoded_seqs.reshape(len(self.encoded_seqs), -1)
        else:
            X = self.encoded_seqs
        
        self.embeddings = self.encoder.predict(X, verbose=0)
        print(f"Extracted embeddings shape: {self.embeddings.shape}")
        return self.embeddings
    
    def cluster_embeddings(self, n_clusters=3, method='kmeans'):
        """Cluster the neural network embeddings"""
        if self.embeddings is None:
            self.extract_embeddings()
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            print(f"Method {method} not implemented, using K-means")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        
        self.labels = clusterer.fit_predict(embeddings_scaled)
        
        # Calculate silhouette score
        if len(set(self.labels)) > 1:
            score = silhouette_score(embeddings_scaled, self.labels)
            print(f"Neural clustering: {n_clusters} clusters, silhouette score: {score:.3f}")
        
        return self.labels
    
    def visualize_embeddings(self):
        """Visualize embeddings and clusters"""
        if self.embeddings is None:
            print("No embeddings available")
            return
        
        # Reduce to 2