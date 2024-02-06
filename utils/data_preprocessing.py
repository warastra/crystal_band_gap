from mp_api.client import MPRester
from typing import List, Dict
import pandas as pd
import numpy as np

# adapted from Harry's introductory notebook
def get_dataset_from_matprojects(elements:List[str]):
    """
    elements filter to download dataset from 
    """

    data = []  
    for e in elements:
        with MPRester('UxYiT0ht7YNg2b6k0oqUVi7LtrUCf9m6') as m:
            temp = m.materials.summary.search(elements=[e])
            data.append(temp)

    formatted_data = [item for sublist in data for item in sublist]
    new_data = []
    for doc in formatted_data:
        new_data.append({
            'material_id': doc.material_id,
            'energy_per_atom': doc.energy_per_atom,
            'structure': doc.structure,
            'composition_dict':doc.composition.as_dict(),
            'formula':doc.formula_pretty,
            'n_elements':doc.nelements,
            'elements':doc.elements,
            'band_gap':doc.band_gap,
            'chemsys':doc.chemsys,
            'comp':doc.composition
        })

    dfC = pd.DataFrame(new_data)
    return dfC

def get_crystal_atom_counts(composition_dict:Dict):
    # Get atom counts
    atom_counts = {}
    for row in composition_dict:
        for key in row.keys():
            atom_counts[key] = atom_counts[key] + row[key] if key in atom_counts.keys() else row[key]
    crystal_atom_counts = pd.Series({ key:atom_counts[key] for key in sorted(atom_counts.keys()) })
    return crystal_atom_counts

def filter_by_element_count(cutoff:int, crystal_atom_counts:Dict, dfC:pd.DataFrame):
    # Only choose structures with elements that are well represented in the dataset
    cutoff = 300
    elements = [key for key in crystal_atom_counts.keys() if crystal_atom_counts[key] >= cutoff]
    print(f"Allowed Elements: {elements}")
    print(f"Number of Elements: {len(elements)}")

    # Lets now filter the structures
    mask = [row[1].name for row in dfC.iterrows() if np.all(np.isin(list(row[1].composition_dict.keys()), elements))]
    filtered = dfC.loc[mask].drop_duplicates(subset='material_id')
    filtered = filtered.loc[filtered.band_gap > 0]
    return filtered

