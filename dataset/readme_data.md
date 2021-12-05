# Note
Feel free to change the path to images in the CSV files (eg in bw_CHIRAL, raw_CHIRAL, bw_CHEMBL+, raw_CHEMBL+).

# Image Naming convention
## CHEMBL+ dataset
{chembl_id}_{database_id}

### Database ID

1: CHEMBL \
2: RDKit \
3: Indigo \
4: Protein Data Bank (PDB) \
5: Chemical Entities of Biological Interest (ChEBI) \
6: Aggregated Computational Toxicology Resource (ACToR) \
7: PubChem \
8: Library of Integrated Network-Based Cellular Signatures (LINCS) \
9: Kyoto Encyclopedia of Genes and Genomes (KEGG Ligand) \
10: ChemSpider \
11: OpenEye

Color
Calculated means: [0.9813, 0.9806, 0.9810]
Calculated stds: [0.0791, 0.0812, 0.0797]

BW
Calculated means:[0.9852, 0.9852, 0.9852]
Calculated stds: [0.1079, 0.1079, 0.1079]

## CHIRAL dataset

{type of chirality}_{molecule assigned id}_{training set id}

### Type of chirality
C: chiral centre \
A: axial chirality \
P: planar chirality \
N: none

### Training set id:
1-9 : training set images \
10: chemdraw images \
12-24: additional training images queried from various databases

Note that:
- AP with molecule assigned id >= 264 are hypothetical molecules
- CAP with molecule assigned id >= 278S are hypothetical molecules
