import random
from Bio import SeqIO

def contains_only_standard_bases(seq):
    return all(base in "ACGTU" for base in seq)

def chunk_sequences_randomly(
    input_fasta,
    output_fasta,
    min_len=100,
    max_len=800,
    target_count=600000
):
    total_chunks = 0
    with open(output_fasta, "w") as out_f:
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq = record.seq.upper()
            seq_len = len(seq)
            pos = 0
            while pos + min_len <= seq_len:
                frag_len = random.randint(min_len, min(max_len, seq_len - pos))
                fragment = seq[pos:pos+frag_len]
                if contains_only_standard_bases(fragment):
                    out_f.write(f">{record.id}_chunk_{pos}-{pos+frag_len}\n{fragment}\n")
                    total_chunks += 1
                pos += frag_len  # non-overlapping
                if total_chunks >= target_count:
                    print(f"\n✅ Reached target: {target_count} sequences written.")
                    return
    print(f"\n✅ Done. Total chunks written: {total_chunks} to {output_fasta}")

# === Run the function ===
if __name__ == "__main__":
    chunk_sequences_randomly(
        input_fasta="top_300_non_erv_sequences.fasta",
        output_fasta="non_erv_chunks_600k.fasta",
        #min_len=100,
        #max_len=400,
        #target_count=600000
    )
