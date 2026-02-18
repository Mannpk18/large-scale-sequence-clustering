from Bio import SeqIO
import gzip
import pandas as pd
from composition_based import CompositionClusterer  # assuming you renamed the file to match import

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

# --- Composition Clustering ---
clusterer = CompositionClusterer()
clusterer.load_sequences(sequences)
clusterer.extract_composition_features()
labels = clusterer.cluster_kmeans(n_clusters=5)

# --- Save Results ---
df = pd.DataFrame({
    "sequence": sequences,
    "composition_cluster": labels
})
df.to_csv("composition_clustered_output.csv", index=False)
print("Saved composition-based clustering output to CSV.")
