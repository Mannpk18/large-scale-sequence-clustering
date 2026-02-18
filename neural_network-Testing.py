from Bio import SeqIO
import gzip
import pandas as pd
from neural_network import NeuralDNAClusterer  # assuming you renamed the file

# --- Load FASTQ sequences ---
def load_sequences(file_path, limit=None):
    sequences = []
    with gzip.open(file_path, "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fastq")):
            sequences.append(str(record.seq))
            if limit and i + 1 >= limit:
                break
    return sequences

file_path = "C:/Users/pinal/Downloads/Testing/5-LD03a_S54_L001_R1_001.fastq.gz"
sequences = load_sequences(file_path, limit=10000)

# --- Neural Net Clustering ---
clusterer = NeuralDNAClusterer()
clusterer.load_sequences(sequences)
clusterer.encode_sequences()
clusterer.build_cnn_autoencoder(encoding_dim=32)
clusterer.train_autoencoder(epochs=10, batch_size=64)
clusterer.extract_embeddings()
labels = clusterer.cluster_embeddings(n_clusters=5)

# --- Save Results ---
df = pd.DataFrame({
    "sequence": sequences,
    "neural_cluster": labels
})
df.to_csv("neural_clustered_output.csv", index=False)
print("Saved neural network clustering output to CSV.")
