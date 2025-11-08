import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import numpy as np
import sys

def generate_html_viewer(smiles, filename="molecule_viewer.html"):

    print(f"1. Loading molecule: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        print("SMILES 문자열로 분자를 만들 수 없습니다.", file=sys.stderr)
        return
    mol = Chem.AddHs(mol)
    
    print("2. Generating 'Local Minimum' 3D structure (MMFF94)...")
    try:
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True 
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol) 
    except Exception as e:
        print(f"Error during 3D embedding or optimization: {e}", file=sys.stderr)
        return

    print("3. Calculating Gasteiger partial charges...")
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e:
        print(f"Gasteiger 전하 계산 실패: {e}", file=sys.stderr)
        return

    print("4. Preparing 3Dmol HTML viewer...")

    mol_block = Chem.MolToPDBBlock(mol)

    v = py3Dmol.view(width=1200, height=800)

    v.addModel(mol_block, 'pdb')

    charges = np.array([float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms() if atom.HasProp('_GasteigerCharge')])
    max_abs_charge = np.max(np.abs(charges))
    if max_abs_charge == 0: max_abs_charge = 0.1 

    v.setStyle({'stick': {'radius': 0.1, 'color': 'lightgray'}, 
                  'sphere': {'radius': 0.3}})


    v.setColorByProperty(
        'atom', 
        'partialCharge', 
        {'prop': '_GasteigerCharge'},
        {'gradient': 'RdBu_r', 'min': -max_abs_charge, 'max': max_abs_charge}
    )


    v.addResLabels(
        'atom', 
        {'font': 'Arial', 'fontSize': 12, 'color': 'black', 'backgroundColor': 'white', 'backgroundOpacity': 0.5}
    )

    v.zoomTo()
    
    v.write_html(filename)

if __name__ == "__main__":
    smiles = "COc1ccc(SC)cc1"
    generate_html_viewer(smiles)