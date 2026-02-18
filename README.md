# Large-Scale Sequence Clustering & Compression

## Overview
This project explores how to efficiently store and process very large biological sequence datasets by clustering similar sequences before compression.

Raw genomic datasets contain massive redundancy.  
Naively storing them wastes both memory and compute.

The goal was to reorganize the data so compression algorithms could operate more effectively.

---

## Problem
Biological sequence files (FASTQ/FASTA) can contain millions of sequences, many of which are highly similar.

Standard compression treats each sequence independently and fails to exploit structural similarity.

The challenge:
How do we group similar sequences together so storage and processing become efficient?

---

## Approach

Pipeline:

Raw Sequences  
→ Feature Extraction (k-mers)  
→ Similarity Measurement  
→ Clustering  
→ Grouped Compression  
→ Compression Evaluation

---

## Key Components

### Feature Extraction
Sequences are converted into k-mer frequency representations to approximate similarity without expensive alignment.

### Clustering
Sequences with similar k-mer profiles are grouped together.  
This improves locality and reduces entropy before compression.

### Compression Evaluation
Compared compression ratios between:
- unclustered sequences
- clustered sequences

---

## Results
Clustering sequences before compression significantly improved compression ratio and reduced redundant storage.

The main performance gains came from data organization rather than the compression algorithm itself.

---

## What This Project Demonstrates
- handling large real-world datasets
- memory-aware processing
- indexing and similarity search
- tradeoffs between compute time and storage efficiency

---

## What I Learned
Efficient systems often come from changing how data is structured, not from using a more complex algorithm.
