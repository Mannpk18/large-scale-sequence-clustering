import pydivsufsort
import gzip
import os
from Bio import SeqIO
import shutil


def load_and_concatenate(fastq_paths):
    sequences = []
    for path in fastq_paths:
        print(f"Reading: {path}")
        with gzip.open(path, "rt") as handle:
            for record in SeqIO.parse(handle, "fastq"):
                sequences.append(str(record.seq) + "$")  
    print(f"Total sequences loaded: {len(sequences)}")
    return "".join(sequences)


def bwt_from_sequence(text):
    print("Generating suffix array...")
    suffix_array = pydivsufsort.divsufsort(text)
    bwt = "".join(text[i - 1] if i != 0 else text[-1] for i in suffix_array)
    return bwt


def save_bwt(bwt_string, out_path):
    with open(out_path, "w") as f:
        f.write(bwt_string)
    print(f"BWT saved to: {out_path}")


def compress_file(input_path, output_path):
    with open(input_path, "rb") as f_in, gzip.open(output_path, "wb", compresslevel=9) as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"Compressed to: {output_path}")


def report_sizes(file_paths):
    for f in file_paths:
        size_kb = os.path.getsize(f) / 1024
        print(f"{f}: {size_kb:.2f} KB")


if __name__ == "__main__":
    fastq_paths = [
        "5-LD03a_S54_L001_R1_001.fastq.gz",
        "5-LD03a_S54_L001_R2_001.fastq.gz"
    ]
    
    output_prefix = "bwt_output"

    
    full_sequence = load_and_concatenate(fastq_paths)

    
    bwt_string = bwt_from_sequence(full_sequence)

    
    txt_file = f"{output_prefix}.txt"
    gz_file = f"{output_prefix}.txt.gz"
    save_bwt(bwt_string, txt_file)
    compress_file(txt_file, gz_file)

    
    report_sizes([txt_file, gz_file])
