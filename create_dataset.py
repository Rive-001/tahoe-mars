import h5py
import numpy as np
import scanpy as sc
import pandas as pd

# # Load AnnData
# adata = sc.read("data/tahoe-uce-embeddings.h5ad")

# # Extract data
# X_uce = adata.obsm["X_uce"]
# cell_ids = adata.obs["Cell_ID_Cellosaur"].to_numpy()
# cell_names = adata.obs["Cell_Name_Vevo"].to_numpy()
# drug = adata.obs["drug"].to_numpy()
# drugname_drugconc = adata.obs["drugname_drugconc"].to_numpy()

# # Define UTF-8 string dtype for h5py
# str_dtype = h5py.string_dtype(encoding='utf-8')

# # Save to HDF5
# with h5py.File("data/all.h5", "w") as f:
#     f.create_dataset("X_uce", data=X_uce)
#     f.create_dataset("cell_ids", data=cell_ids, dtype=str_dtype)
#     f.create_dataset("cell_names", data=cell_names, dtype=str_dtype)
#     f.create_dataset("drug", data=drug, dtype=str_dtype)
#     f.create_dataset("drugname_drugconc", data=drugname_drugconc, dtype=str_dtype)

# print("Saved data to data/all.h5 with UTF-8 encoding")

# # Step 1: Load cell_ids from all.h5
# with h5py.File("data/all.h5", "r") as f:
#     cell_ids_all = f["cell_ids"][:].astype(str)  # convert from bytes to str

# # print(cell_ids_all.shape)


# # Step 2: Load vision scores AnnData
# adata_vision = sc.read("data/tahoe-vision-scores.h5ad")

# vision_ids = adata_vision.obs["Cell_ID_Cellosaur"].to_numpy()
# X_vision = adata_vision.X
# X_vision = X_vision.toarray() if hasattr(X_vision, "toarray") else X_vision  # ensure dense

# # Step 3: Create index mapping using NumPy
# # Build a dictionary from vision cell ID to row index
# vision_id_to_index = {id_: i for i, id_ in enumerate(vision_ids)}

# # Get indices of cell_ids_all in vision_ids
# try:
#     indices = np.array([vision_id_to_index[id_] for id_ in cell_ids_all])
# except KeyError as e:
#     missing = set(cell_ids_all) - set(vision_id_to_index)
#     raise ValueError(f"Missing Cell_IDs in vision data: {missing}")

# # Step 4: Reorder vision scores
# vision_scores_aligned = X_vision[indices]

# # Step 5: Save to HDF5 file
# with h5py.File("data/all.h5", "a") as f:
#     if "vision_scores" in f:
#         del f["vision_scores"]  # remove old dataset if exists
#     f.create_dataset("vision_scores", data=vision_scores_aligned)

# print("✅ Vision scores aligned and saved to data/all.h5")

# Open the HDF5 file in read mode
with h5py.File("data/all.h5", "r") as f:
    drugs = f["drug"][:]

drugs = [
    d.decode("utf-8").strip()              # decode bytes → str, then strip
    if isinstance(d, (bytes, bytearray)) 
    else str(d).strip()                    # also ensure any str is stripped
    for d in drugs
]
# Load ChemBERTA embeddings CSV
df = pd.read_csv("data/ChemBert_drug_embeddings_375.csv")  # Adjust the filename as needed

# Display available columns
print("Columns in CSV:", df.columns.tolist())

# Extract relevant columns
drug_names = df["drug"].to_numpy()
smiles = df["canonical_smiles"].to_numpy()

drug2emb = dict(zip(drug_names, smiles))

# 2. Get the unique set of drug names from HDF5
set_hdf5 = set(np.unique(drugs))

# 3. Get the set of drug names in your embedding dict
set_csv = set(drug2emb.keys())

# 4. Compare!
print(f"# unique in HDF5:        {len(set_hdf5)}")
print(f"# in embedding dict:    {len(set_csv)}")
print("Sets equal? →", set_hdf5 == set_csv)

# 5. If they’re not equal, find the differences:
print("\nIn CSV but not in HDF5:", set_csv - set_hdf5)
print("\nIn HDF5 but not in CSV:", set_hdf5 - set_csv)

drug_embeddings = []

for drug in drugs:
	try:
		drug_embeddings.append(drug2emb[drug])
	except:
		drug_embeddings.append(np.zeros[784])
