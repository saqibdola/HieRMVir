import os
import re
import csv
import subprocess
from pathlib import Path
import pandas as pd
import random
import numpy as np

# === Fixed seeds for reproducibility ===
random.seed(42)
np.random.seed(42)

# === IUPAC ambiguity codes and resolver ===
IUPAC_CODES = {
    'A': ['A'],
    'C': ['C'],
    'G': ['G'],
    'T': ['T'],
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'B': ['C', 'G', 'T'],
    'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G'],
    'N': ['A', 'C', 'G', 'T']
}

def resolve_ambiguous_bases(seq):
    """Replace ambiguous IUPAC codes with a random valid nucleotide."""
    return ''.join(random.choice(IUPAC_CODES.get(base, ['N'])) for base in seq.upper())


# === Subprocess Output Handling ===
def run_and_capture(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding='utf8', errors='replace')
    full_output = ""
    for line in process.stdout:
        print(line, end="")  # Live output
        full_output += line
    process.wait()
    return full_output


def parse_metrics(output_text):
    acc = f1 = prec = rec = h_prec = h_rec = h_f1 = "N/A"
    for line in output_text.splitlines():
        if "Accuracy:" in line:
            acc = line.split("Accuracy:")[1].strip()
        elif "Precision:" in line and "Hierarchical" not in line:
            prec = line.split("Precision:")[1].strip()
        elif "Recall:" in line and "Hierarchical" not in line:
            rec = line.split("Recall:")[1].strip()
        elif "F1 Score:" in line and "Hierarchical" not in line:
            f1 = line.split("F1 Score:")[1].strip()
        elif "Hierarchical Precision:" in line:
            h_prec = line.split("Hierarchical Precision:")[1].strip()
        elif "Hierarchical Recall:" in line:
            h_rec = line.split("Hierarchical Recall:")[1].strip()
        elif "Hierarchical F1 Score:" in line:
            h_f1 = line.split("Hierarchical F1 Score:")[1].strip()
    return acc, prec, rec, f1, h_prec, h_rec, h_f1



# === Fasta Parsing and Labeling ===
def fasta_to_processed(input_file, output_file, min_length=100):
    bad_lines = []
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        seq = ''
        for line in infile:
            if line.startswith('>'):
                if seq and len(seq) >= min_length:
                    resolved = resolve_ambiguous_bases(seq)
                    outfile.write(resolved + '\n')
                seq = ''
            else:
                seq += line.strip()
        if seq and len(seq) >= min_length:
            resolved = resolve_ambiguous_bases(seq)
            outfile.write(resolved + '\n')

def label_sequences(input_files, labels, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sequence', 'Label'])
        for fpath, label in zip(input_files, labels):
            with open(fpath, 'r') as f:
                for line in f:
                    seq = line.strip()
                    if seq:
                        writer.writerow([seq, label])
    print(f"✅ Wrote labeled sequences to {output_file}")

def extract_files(folder, exclude_erv=True):
    files = []
    labels = []
    for fname in os.listdir(folder):
        if "_processed" not in fname:
            continue
        if exclude_erv and ("erv" in fname.lower() or "nonerv" in fname.lower()):
            continue

        path = os.path.join(folder, fname)
        fname_lower = fname.lower()

        if "nonerv" in fname_lower:
            label = "NONERV"
        elif "erv" in fname_lower:
            label = "ERV"
        else:
            label = "Virus"

        files.append(path)
        labels.append(label)

    return files, labels

def extract_files_level2(folder):
    files = []
    labels = []
    for fname in os.listdir(folder):
        if "_processed" not in fname:
            continue
        if "erv" in fname.lower() or "nonerv" in fname.lower():
            continue  # exclude non-viral

        path = os.path.join(folder, fname)
        match = re.search(r'C[1-7]', fname)
        if match:
            label = match.group(0)
            files.append(path)
            labels.append(label)
        else:
            print(f"⚠️ Skipping {fname}: No C[1-7] match")

    return files, labels

# === Step 1: Label Level 1 and Level 2 ===
print("🔹 Generating labeled_level1.csv...")
l1_files, l1_labels = extract_files("BaltimoreClassification", exclude_erv=False)
label_sequences(l1_files, l1_labels, "labeled_level1.csv")

print("🔹 Generating labeled_level2.csv...")
l2_files, l2_labels = extract_files_level2("BaltimoreClassification")
label_sequences(l2_files, l2_labels, "labeled_level2.csv")


# === Step 2: Label Species-Level ===
print("🔧 Preprocessing and labeling species FASTA files for Level 3...")
species_input = "BaltimoreClassification/Species"
species_output_root = "species_by_class"
Path(species_output_root).mkdir(exist_ok=True)

for file in os.listdir(species_input):
    if not file.endswith(('.fasta', '.fa')):
        continue
    baltimore = re.search(r'C[1-7]', file)
    if not baltimore:
        print(f"⚠️ Skipping {file}: No Baltimore class found")
        continue
    class_label = baltimore.group(0)
    species_name = re.sub(r'C[1-7]', '', file).replace('.fasta', '').replace('.fa', '').strip().replace(' ', '_')
    output_dir = os.path.join(species_output_root, class_label)
    Path(output_dir).mkdir(exist_ok=True)
    input_path = os.path.join(species_input, file)
    output_path = os.path.join(output_dir, f"{species_name}_processed.txt")
    fasta_to_processed(input_path, output_path)

print("🔹 Labeling species-level CSVs...")
for class_dir in sorted(os.listdir(species_output_root)):
    class_path = os.path.join(species_output_root, class_dir)
    if not os.path.isdir(class_path) or not re.fullmatch(r'C[1-7]', class_dir):
        continue

    species_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith("_processed.txt")]
    species_labels = [f.split("_processed")[0] for f in os.listdir(class_path) if f.endswith("_processed.txt")]

    output_csv = f"labeled_{class_dir}.csv"
    label_sequences(species_files, species_labels, output_csv)

# === Step 3: Run Pipeline for k = 3 to 5 ===
results = []

for k in range(3, 6):
    print(f"\n🔁 === Running Level 1 for k={k} ===")
    run_and_capture(["python", "step2withchunks4hierichial.py", "--input", "labeled_level1.csv", "--output", f"kmer_level1_k{k}.csv", "--k", str(k)])
    run_and_capture(["python", "step3newwithoutchunking4hier.py", "--input", f"kmer_level1_k{k}.csv", "--output", f"scaled_level1_k{k}.csv", "--importance_out", f"importance_level1_k{k}.csv"])
    l1_result = run_and_capture(["python", "step4withoutchunking optimization23june.py", "--input", f"scaled_level1_k{k}.csv", "--model_out", f"model_level1_k{k}.pt", "--encoder_out", f"encoder_level1_k{k}.pkl", "--loss_out", f"loss_level1_k{k}.csv"])
    acc1, prec1, rec1, f1_1, hprec1, hrec1, hf1_1 = parse_metrics(l1_result)
    results.append({
        "k": k, "Level": "Level 1",
        "Accuracy": acc1,
        "Precision": prec1,
        "Recall": rec1,
        "F1 Score": f1_1,
        "Hierarchical Precision": hprec1,
        "Hierarchical Recall": hrec1,
        "Hierarchical F1 Score": hf1_1
    })

    print(f"\n🔁 === Running Level 2 for k={k} ===")
    run_and_capture(["python", "step2withchunks4hierichial.py", "--input", "labeled_level2.csv", "--output", f"kmer_level2_k{k}.csv", "--k", str(k)])
    run_and_capture(["python", "step3newwithoutchunking4hier.py", "--input", f"kmer_level2_k{k}.csv", "--output", f"scaled_level2_k{k}.csv", "--importance_out", f"importance_level2_k{k}.csv"])
    l2_result = run_and_capture(["python", "step4withoutchunking optimization23june.py", "--input", f"scaled_level2_k{k}.csv", "--model_out", f"model_level2_k{k}.pt", "--encoder_out", f"encoder_level2_k{k}.pkl", "--loss_out", f"loss_level2_k{k}.csv"])
    acc2, prec2, rec2, f1_2, hprec2, hrec2, hf1_2 = parse_metrics(l2_result)
    results.append({
        "k": k, "Level": "Level 2",
        "Accuracy": acc2,
        "Precision": prec2,
        "Recall": rec2,
        "F1 Score": f1_2,
        "Hierarchical Precision": hprec2,
        "Hierarchical Recall": hrec2,
        "Hierarchical F1 Score": hf1_2
    })

    print(f"\n🔁 === Running Level 3 (species) for k={k} ===")
    for c in range(1, 8):
        class_label = f"C{c}"
        labeled = f"labeled_{class_label}.csv"
        kmer_out = f"kmer_{class_label}_k{k}.csv"
        scaled_out = f"scaled_{class_label}_k{k}.csv"
        imp_out = f"importance_{class_label}_k{k}.csv"
        model_out = f"model_{class_label}_k{k}.pt"
        enc_out = f"encoder_{class_label}_k{k}.pkl"

        run_and_capture(["python", "step2withchunks4hierichial.py", "--input", labeled, "--output", kmer_out, "--k", str(k)])
        run_and_capture(["python", "step3newwithoutchunking4hier.py", "--input", kmer_out, "--output", scaled_out, "--importance_out", imp_out])
        l3_result = run_and_capture(["python", "step4withoutchunking optimization23june.py", "--input", scaled_out, "--model_out", model_out, "--encoder_out", enc_out,  "--loss_out", f"loss_{class_label}_k{k}.csv"])
        acc3, prec3, rec3, f1_3, hprec3, hrec3, hf1_3 = parse_metrics(l3_result)
        results.append({
            "k": k, "Level": "Level 3",
            "Accuracy": acc3,
            "Precision": prec3,
            "Recall": rec3,
            "F1 Score": f1_3,
            "Hierarchical Precision": hprec3,
            "Hierarchical Recall": hrec3,
            "Hierarchical F1 Score": hf1_3
        })


# === Save Results ===
df = pd.DataFrame(results)
df.to_csv("kmer_results.csv", index=False)
print("\n📄 Saved all results to kmer_results.csv")
