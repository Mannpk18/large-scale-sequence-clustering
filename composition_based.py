#!/usr/bin/env python3
"""
Composition-Based DNA Sequence Clustering
Uses nucleotide composition and derived features for clustering
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

class CompositionClusterer:
    def __init__(self):
        self.sequences = []
        self.features = None
        self.labels = None
        self.feature_names = []
        
    def load_sequences(self, sequences):
        """Load DNA sequences"""
        self.sequences = [seq.upper() for seq in sequences]
        return self
        
    def load_fasta(self, filename):
        """Load sequences from FASTA file"""
        sequences = []
        current_seq = ""
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
        
        self.sequences = [seq.upper() for seq in sequences]
        print(f"Loaded {len(self.sequences)} sequences")
        return self
    
    def extract_composition_features(self):
        """Extract comprehensive composition features"""
        print("Extracting composition features...")
        
        features = []
        feature_names = []
        
        for seq in self.sequences:
            seq_len = len(seq)
            if seq_len == 0:
                features.append([0] * 12)
                continue
            
            # Basic nucleotide frequencies
            a_freq = seq.count('A') / seq_len
            t_freq = seq.count('T') / seq_len
            g_freq = seq.count('G') / seq_len
            c_freq = seq.count('C') / seq_len
            
            # Derived composition features
            gc_content = g_freq + c_freq
            at_content = a_freq + t_freq
            purine_content = a_freq + g_freq  # A, G are purines
            pyrimidine_content = t_freq + c_freq  # T, C are pyrimidines
            
            # Skew measures
            gc_skew = (g_freq - c_freq) / (g_freq + c_freq) if (g_freq + c_freq) > 0 else 0
            at_skew = (a_freq - t_freq) / (a_freq + t_freq) if (a_freq + t_freq) > 0 else 0
            
            # Additional features
            sequence_length = seq_len
            entropy = self._calculate_entropy([a_freq, t_freq, g_freq, c_freq])
            
            feature_vector = [
                a_freq, t_freq, g_freq, c_freq,
                gc_content, at_content, purine_content, pyrimidine_content,
                gc_skew, at_skew, sequence_length, entropy
            ]
            
            features.append(feature_vector)
        
        self.features = np.array(features)
        self.feature_names = [
            'A_freq', 'T_freq', 'G_freq', 'C_freq',
            'GC_content', 'AT_content', 'Purine_content', 'Pyrimidine_content',
            'GC_skew', 'AT_skew', 'Length', 'Entropy'
        ]
        
        print(f"Extracted {self.features.shape[1]} composition features")
        return self.features
    
    def _calculate_entropy(self, freqs):
        """Calculate Shannon entropy of nucleotide frequencies"""
        entropy = 0
        for freq in freqs:
            if freq > 0:
                entropy -= freq * np.log2(freq)
        return entropy
    
    def extract_dinucleotide_features(self):
        """Extract dinucleotide composition features"""
        print("Extracting dinucleotide features...")
        
        bases = ['A', 'T', 'G', 'C']
        dinucs = [b1 + b2 for b1 in bases for b2 in bases]
        
        features = []
        for seq in self.sequences:
            dinuc_counts = {dinuc: 0 for dinuc in dinucs}
            
            # Count dinucleotides
            for i in range(len(seq) - 1):
                dinuc = seq[i:i+2]
                if dinuc in dinuc_counts:
                    dinuc_counts[dinuc] += 1
            
            # Convert to frequencies
            total = sum(dinuc_counts.values())
            feature_vector = []
            for dinuc in dinucs:
                freq = dinuc_counts[dinuc] / total if total > 0 else 0
                feature_vector.append(freq)
            
            features.append(feature_vector)
        
        self.features = np.array(features)
        self.feature_names = dinucs
        print(f"Extracted {len(dinucs)} dinucleotide features")
        return self.features
    
    def cluster_kmeans(self, n_clusters=3):
        """Perform K-means clustering"""
        if self.features is None:
            self.extract_composition_features()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = kmeans.fit_predict(features_scaled)
        
        # Calculate metrics
        if len(set(self.labels)) > 1:
            score = silhouette_score(features_scaled, self.labels)
            print(f"K-means clustering: {n_clusters} clusters, silhouette score: {score:.3f}")
        
        return self.labels
    
    def cluster_hierarchical(self, n_clusters=3, linkage='ward'):
        """Perform hierarchical clustering"""
        if self.features is None:
            self.extract_composition_features()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # Cluster
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage
        )
        self.labels = hierarchical.fit_predict(features_scaled)
        
        print(f"Hierarchical clustering: {n_clusters} clusters")
        return self.labels
    
    def analyze_feature_importance(self):
        """Analyze which features are most important for clustering"""
        if self.features is None or self.labels is None:
            print("No clustering results available")
            return
        
        # Calculate feature means for each cluster
        unique_labels = sorted(set(self.labels))
        cluster_means = []
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(self.labels) if l == label]
            cluster_features = self.features[cluster_indices]
            cluster_means.append(np.mean(cluster_features, axis=0))
        
        cluster_means = np.array(cluster_means)
        
        # Calculate feature variance across clusters
        feature_variances = np.var(cluster_means, axis=0)
        
        # Sort features by importance
        importance_order = np.argsort(feature_variances)[::-1]
        
        print("\nFeature importance (by variance across clusters):")
        for i, idx in enumerate(importance_order[:5]):
            if idx < len(self.feature_names):
                print(f"{i+1}. {self.feature_names[idx]}: {feature_variances[idx]:.4f}")
    
    def visualize_clusters(self):
        """Visualize clusters using PCA"""
        if self.features is None or self.labels is None:
            print("No clustering results to visualize")
            return
        
        # Standardize and reduce dimensions
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        # PCA plot
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=self.labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA of Composition Features')
        
        # Feature heatmap if we have named features
        if len(self.feature_names) <= 20:  # Only if not too many features
            plt.subplot(1, 2, 2)
            
            # Calculate cluster centroids
            unique_labels = sorted(set(self.labels))
            centroids = []
            for label in unique_labels:
                cluster_indices = [i for i, l in enumerate(self.labels) if l == label]
                centroid = np.mean(self.features[cluster_indices], axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            # Create heatmap
            sns.heatmap(centroids.T, 
                       xticklabels=[f'Cluster {i}' for i in unique_labels],
                       yticklabels=self.feature_names,
                       cmap='viridis', cbar=True)
            plt.title('Cluster Centroids')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def get_cluster_stats(self):
        """Get detailed statistics for each cluster"""
        if self.labels is None:
            return None
        
        stats = {}
        unique_labels = sorted(set(self.labels))
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(self.labels) if l == label]
            cluster_seqs = [self.sequences[i] for i in cluster_indices]
            cluster_features = self.features[cluster_indices]
            
            stats[label] = {
                'count': len(cluster_indices),
                'avg_length': np.mean([len(seq) for seq in cluster_seqs]),
                'std_length': np.std([len(seq) for seq in cluster_seqs]),
                'feature_means': np.mean(cluster_features, axis=0),
                'feature_stds': np.std(cluster_features, axis=0),
                'sample_indices': cluster_indices[:3]  # First 3 examples
            }
        
        return stats

# Example usage
if __name__ == "__main__":
    # Generate sample sequences with different compositions
    sequences = [
        "AAAAAATTTTTTTAAAAAATTTTTTT",  # AT-rich
        "AAAAAATTTTTTAAAAAAATTTTTT",   # AT-rich similar
        "GGGGGGCCCCCCCGGGGGGCCCCCC",   # GC-rich
        "GGGGGCCCCCCGGGGGCCCCCC",      # GC-rich similar
        "ATGCATGCATGCATGCATGCATGC",    # Balanced
        "ATGCATGCATGCATGCATGCATGCA",   # Balanced similar
    ]
    
    # Initialize and run clustering
    clusterer = CompositionClusterer()
    clusterer.load_sequences(sequences)
    
    # Try composition features
    features = clusterer.extract_composition_features()
    labels = clusterer.cluster_kmeans(n_clusters=3)
    
    print("Composition-based clustering results:")
    print("Labels:", labels)
    
    # Analyze results
    clusterer.analyze_feature_importance()
    print("\nCluster statistics:")
    stats = clusterer.get_cluster_stats()
    for label, stat in stats.items():
        print(f"Cluster {label}: {stat['count']} sequences, "
              f"avg length: {stat['avg_length']:.1f}")
    
    # Try dinucleotide features
    clusterer.extract_dinucleotide_features()
    dinuc_labels = clusterer.cluster_kmeans(n_clusters=3)
    print("\nDinucleotide-based labels:", dinuc_labels)