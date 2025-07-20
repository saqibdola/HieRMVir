import pandas as pd
from collections import Counter
import itertools
from tqdm import tqdm
import numpy as np
import os
import argparse


# === PARSE COMMAND LINE ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input CSV file with Sequence and Label columns")
parser.add_argument("--output", required=True, help="Output CSV file for k-mer features")
parser.add_argument("--k", type=int, default=5, help="Length of k-mers")
args = parser.parse_args()
print(f"📥 step2kmerwithchunks.py → Input file: {args.input}, k-mer: {args.k}, Output: {args.output}")


input_file = args.input
output_file = args.output
k = args.k


chunk_size = 100000  # Process N sequences at a time
write_header = True  # Only write header on first chunk

# === Step 1: Generate all possible k-mers ===
alphabet = ['A', 'C', 'G', 'T']
all_kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]

# === Step 2: K-mer frequency function ===
def compute_kmer_freq(sequence, k, kmer_list):
    sequence = str(sequence).upper()
    kmer_counts = Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))
    total_kmers = sum(kmer_counts.values())
    return [kmer_counts[kmer] / total_kmers if total_kmers > 0 else 0.0 for kmer in kmer_list]

# === Step 3: Process in Chunks ===
def process_chunk(chunk, k, kmer_list):
    tqdm.pandas(desc="K-mer Processing")
    kmer_vectors = chunk['Sequence'].progress_apply(lambda seq: compute_kmer_freq(seq, k, kmer_list))
    kmer_df = pd.DataFrame(kmer_vectors.tolist(), columns=kmer_list)
    kmer_df['Label'] = chunk['Label'].values
    return kmer_df

# === Step 4: Process and Write in Batches ===
if os.path.exists(output_file):
    os.remove(output_file)  # Clean existing output

with pd.read_csv(input_file, chunksize=chunk_size) as reader:
    for i, chunk in enumerate(reader):
        print(f"\n🔄 Processing chunk {i+1}...")
        chunk = chunk.dropna(subset=["Sequence", "Label"])
        chunk['Sequence'] = chunk['Sequence'].astype(str)

        processed = process_chunk(chunk, k, all_kmers)

        # Append to output CSV
        processed.to_csv(output_file, mode='a', index=False, header=write_header)
        write_header = False

print(f"\n✅ Finished. Output saved to: {output_file}")
