from Bio import SeqIO
from kmer import KmerClusterer
import pandas as pd
import os
import gzip
import shutil

# === 1. Load Sequences ===
def load_sequences(file_paths, limit=None):
    sequences = []
    total = 0
    for file_path in file_paths:
        with gzip.open(file_path, "rt") as handle:
            for record in SeqIO.parse(handle, "fastq"):
                sequences.append(str(record.seq))
                total += 1
                if limit and total >= limit:
                    return sequences
    return sequences


file_paths = [
    "C:/Users/pinal/Downloads/Testing/5-LD03a_S54_L001_R1_001.fastq.gz",
    "C:/Users/pinal/Downloads/Testing/5-LD03a_S54_L001_R2_001.fastq.gz"  
]

sequences = load_sequences(file_paths)

# === 2. Cluster Using k-mer ===
k = 6
num_clusters = 30
clusterer = KmerClusterer(k=k)
clusterer.load_sequences(sequences)
clusterer.extract_kmer_features()
labels = clusterer.cluster_kmeans(n_clusters=num_clusters)

df = pd.DataFrame({
    "sequence": sequences,
    "cluster": labels
})
df.to_csv("kmer_clustered_output.csv", index=False)

# === 3. Save Files ===
with open("unclustered_sequences.txt", "w") as f:
    for seq in sequences:
        f.write(seq + "\n")

df_sorted = df.sort_values("cluster")
with open("clustered_sequences.txt", "w") as f:
    for seq in df_sorted["sequence"]:
        f.write(seq + "\n")

# === 4. Compress Files ===
def compress_file(input_file, output_file):
    with open(input_file, 'rb') as f_in, gzip.open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

compress_file("unclustered_sequences.txt", "unclustered_sequences.txt.gz")
compress_file("clustered_sequences.txt", "clustered_sequences.txt.gz")

# === 5. Report Sizes ===
def get_size(path):
    return os.path.getsize(path) / 1024  # in KB

print("📦 Compression Summary:")
print(f"Unclustered Compressed Size: {get_size('unclustered_sequences.txt.gz'):.2f} KB")
print(f"Clustered Compressed Size:   {get_size('clustered_sequences.txt.gz'):.2f} KB")


