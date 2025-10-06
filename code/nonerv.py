from Bio import SeqIO
from collections import defaultdict

def load_erv_regions(mask_file):
    erv_coords = defaultdict(list)
    with open(mask_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("SW", "#", "score", "position")):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                chrom = parts[4]
                start = int(parts[5])
                end = int(parts[6])
                family = parts[10]
                name = parts[9]
            except ValueError:
                continue
            if "ERV" in family or "HERV" in name:
                erv_coords[chrom].append((start, end))
    return erv_coords

def get_non_erv_intervals(seq_len, erv_list):
    erv_list.sort()
    non_ervs = []
    prev_end = 0
    for start, end in erv_list:
        if start > prev_end:
            non_ervs.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < seq_len:
        non_ervs.append((prev_end, seq_len))
    return non_ervs

def contains_only_standard_bases(seq):
    return all(base in "ACGTU" for base in seq)

def extract_top_300_non_erv_sequences(fasta_file, mask_file, output_fasta, min_len=100, output_lengths="top_300_non_erv_lengths.txt"):
    erv_regions = load_erv_regions(mask_file)
    candidate_sequences = []

    print("🔍 Scanning for non-ERV regions...")

    for record in SeqIO.parse(fasta_file, "fasta"):
        chrom = record.id
        seq = record.seq
        seq_len = len(seq)

        erv_list = erv_regions.get(chrom, [])
        non_ervs = get_non_erv_intervals(seq_len, erv_list)

        for i, (start, end) in enumerate(non_ervs):
            fragment = seq[start:end].upper()
            if len(fragment) >= min_len and contains_only_standard_bases(fragment):
                seq_id = f"{chrom}_nonERV_{i}_{start}-{end}"
                candidate_sequences.append((seq_id, fragment))

    print(f"✅ Found {len(candidate_sequences)} clean non-ERV fragments.")
    print(f"📏 Sorting and selecting top 300 by length...")

    candidate_sequences.sort(key=lambda x: len(x[1]), reverse=True)
    top_300 = candidate_sequences[:600]

    with open(output_fasta, "w") as out_f, open(output_lengths, "w") as len_f:
        for seq_id, fragment in top_300:
            out_f.write(f">{seq_id}\n{fragment}\n")
            len_f.write(f"{seq_id}\t{len(fragment)}\n")

    print(f"\n✅ Done. Total sequences written: {len(top_300)}")
    print(f"📁 FASTA output: {output_fasta}")
    print(f"📄 Lengths saved in: {output_lengths}")

# === Run the function ===
if __name__ == "__main__":
    extract_top_300_non_erv_sequences(
        fasta_file="GCF_000001405.39_GRCh38.p13_genomic.fna",
        mask_file="hg38.sorted.fa.out",
        output_fasta="top_300_non_erv_sequences.fasta",
        min_len=100,
        output_lengths="top_300_non_erv_lengths.txt"
    )
