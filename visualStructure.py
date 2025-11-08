from rdkit import Chem
from rdkit.Chem import AllChem
import pyvista as pv
import numpy as np

def view_molecule_in_3d(smiles):

    print(f"1. Loading molecule: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    print("2. Generating 3D structure (MMFF94)...")
    try:
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    except (RuntimeError, ValueError) as e:
        print(f"Error during 3D embedding: {e}")
        return

    print("3. Calculating Gasteiger partial charges...")
    AllChem.ComputeGasteigerCharges(mol)
    
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    charges = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()])
    atom_labels = [atom.GetSymbol() for atom in mol.GetAtoms()]

    print("4. Preparing 3D visualization with PyVista...")
    
    points = pv.PolyData(coords)
    points['charge'] = charges
    
    atom_spheres = points.glyph(
        geom=pv.Sphere(radius=0.3),
        scale=False,
        orient=False
    )
    
    lines = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        lines.append([2, idx1, idx2]) 

    bond_lines_mesh = pv.PolyData(coords, lines=np.hstack(lines))
    bond_tubes = bond_lines_mesh.tube(radius=0.08)

    print("5. Launching 3D viewer window...")
    plotter = pv.Plotter(window_size=[900, 700])
    
    plotter.add_mesh(
        atom_spheres,
        scalars='charge',      
        cmap='RdBu_r',         
        scalar_bar_args={'title': 'Partial Charge (Gasteiger)'} 
    )

    plotter.add_mesh(bond_tubes, color='lightgrey')
    max_abs_charge = np.max(np.abs(charges))
    if max_abs_charge == 0:
        max_abs_charge = 0.1 

    plotter.add_mesh(
        atom_spheres,
        scalars='charge',      
        cmap='RdBu_r', 
        clim=[-max_abs_charge, max_abs_charge], 
        scalar_bar_args={'title': 'Partial Charge (Gasteiger)'} 
    )
    plotter.add_point_labels(
        coords,                 
        atom_labels,            
        font_size=12,
        text_color='black',     
        shape=None,             
        show_points=False,
        always_visible=True
    )
    
    plotter.background_color = 'lightgrey' 
    
    plotter.add_text("3D Molecule View - Partial Charges", font_size=15)

    plotter.show()
    
    print("Done. PyVista window closed.")

if __name__ == "__main__":
    smiles = "COc1ccc(SC)cc1"
    view_molecule_in_3d(smiles)