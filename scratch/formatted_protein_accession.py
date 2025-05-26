import pandas as pd

# Load the downloaded TSV file
infile = "human_reviewed_proteins.tsv"
df = pd.read_csv(infile, sep="\t")

# Robust renaming and column selection
rename_map = {}
if "Entry" in df.columns:
    rename_map["Entry"] = "UniProtKB Accession"
if "Protein names" in df.columns:
    rename_map["Protein names"] = "Protein Name"
if "Organism" in df.columns:
    rename_map["Organism"] = "Organism"
if "Organism name" in df.columns:
    rename_map["Organism name"] = "Organism"
if "accession" in df.columns:
    rename_map["accession"] = "UniProtKB Accession"
if "protein_name" in df.columns:
    rename_map["protein_name"] = "Protein Name"
if "organism_name" in df.columns:
    rename_map["organism_name"] = "Organism"

df = df.rename(columns=rename_map)

# Only keep the required columns
required_cols = ["Protein Name", "Organism", "UniProtKB Accession"]
df = df[[col for col in required_cols if col in df.columns]]

# Save to new TSV file
outfile = "formatted_proteins.tsv"
df.to_csv(outfile, sep="\t", index=False)

print(f"✅ File saved as '{outfile}' with {len(df)} proteins.")
