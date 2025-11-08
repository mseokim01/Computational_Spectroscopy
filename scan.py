import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem import AllChem


calToJ = 4.184

def get_constrained_energy_kj(mol, constraints):

        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        
        for constraint in constraints:

            angle_float = float(constraint['angle'])

            ff.UFFAddTorsionConstraint(
                constraint['indices'][0],
                constraint['indices'][1],
                constraint['indices'][2],
                constraint['indices'][3],
                False,
                angle_float,
                angle_float,
                constraint['force_constant_kcal']
            )
        

        ret_code = ff.Minimize(maxIts=1000) 
        
        if ret_code == 1:
            print(f"      > 경고: (S={constraints[0]['angle']}°, O={constraints[1]['angle']}°) 최소화 수렴 실패. (NaN 반환)")
            return np.nan

        energy_kcal = ff.CalcEnergy()
        return energy_kcal * calToJ

print("--- 2D Relaxed Potential Energy Scan (MMFF94) ---")
print("논문(Fig. 3)과 동일한 2D 스캔을 시작합니다.")

smiles = "COc1ccc(SC)cc1"

scan_angles_deg = np.arange(0, 361, 15)
num_steps = len(scan_angles_deg)
print(f"스캔 그리드: {num_steps} x {num_steps} = {num_steps*num_steps} points")

mol_base = Chem.AddHs(Chem.MolFromSmiles(smiles))
AllChem.EmbedMolecule(mol_base)
AllChem.MMFFOptimizeMolecule(mol_base)

smarts_o = "[CH3:1]-[O:2]-[c:3]:[c:4]" 
smarts_s = "[CH3:1]-[S:2]-[c:3]:[c:4]" 

pattern_o = Chem.MolFromSmarts(smarts_o)
pattern_s = Chem.MolFromSmarts(smarts_s)
indices_o = mol_base.GetSubstructMatch(pattern_o)
indices_s = mol_base.GetSubstructMatch(pattern_s)

if not (indices_o and indices_s):
    print("오류: 분자에서 SMARTS 패턴을 찾지 못했습니다.", file=sys.stderr)
    sys.exit(1)

energy_grid = np.zeros((num_steps, num_steps))


force_constant_kcal = 1000.0 

for i, angle_s in enumerate(scan_angles_deg):
    for j, angle_o in enumerate(scan_angles_deg):
        print(f"  Calculating (S={angle_s}°, O={angle_o}°) ...")
        
        mol_step = Chem.Mol(mol_base) 
        
        constraints = [
            { 
                'indices': indices_s,
                'angle': angle_s,
                'force_constant_kcal': force_constant_kcal
            },
            { 
                'indices': indices_o,
                'angle': angle_o,
                'force_constant_kcal': force_constant_kcal
            }
        ]
        
        energy = get_constrained_energy_kj(mol_step, constraints)
        energy_grid[i, j] = energy 

print("2D 스캔 완료. 플로팅 시작...")


try:
    min_energy = np.nanmin(energy_grid)
except ValueError:
    print("오류: 모든 계산이 실패하여(NaN) 최소값을 찾을 수 없습니다.", file=sys.stderr)
    sys.exit(1)

relative_energy_grid = energy_grid - min_energy

VIZ_MAX_ENERGY = 15

clipped_energy_grid = np.nan_to_num(relative_energy_grid, nan=VIZ_MAX_ENERGY)
clipped_energy_grid = np.clip(clipped_energy_grid, 0, VIZ_MAX_ENERGY)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

Y_o, X_s = np.meshgrid(scan_angles_deg, scan_angles_deg)

levels = np.linspace(0, VIZ_MAX_ENERGY, 11) 
cmap = plt.get_cmap('jet')


surf = ax.plot_surface(X_s, Y_o, clipped_energy_grid.T, cmap=cmap, 
                       vmax=VIZ_MAX_ENERGY, vmin=0,
                       edgecolor='none', rstride=1, cstride=1)


cset = ax.contourf(X_s, Y_o, clipped_energy_grid.T, 
                   levels=levels, 
                   zdir='z', offset=-2, 
                   cmap=cmap, alpha=0.8) 

cset_top = ax.contourf(X_s, Y_o, clipped_energy_grid.T, 
                       levels=levels, 
                       zdir='z', offset=20, 
                       cmap=cmap, alpha=0.5)


ax.set_title(f"2D PES Scan (MMFF94) - Clipped at {VIZ_MAX_ENERGY} kJ/mol", fontsize=14)


ax.set_xlabel("$\phi$ C5-C4-O16-C17", fontsize=10, labelpad=10)
ax.set_ylabel("$\phi$ C12-S11-C1-C5", fontsize=10, labelpad=10)
ax.set_zlabel("Relative Energy [kJ/mol]", fontsize=10, labelpad=10)

ax.set_xticks(np.arange(0, 361, 45))
ax.set_yticks(np.arange(0, 361, 45))
ax.set_zlim(0, VIZ_MAX_ENERGY)
ax.set_zticks(np.arange(-2, 21, 2))

cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1, label="Energy [kJ/mol]")
cbar.set_ticks(np.arange(0, VIZ_MAX_ENERGY + 1, 3))

ax.view_init(elev=25, azim=225) 
plt.show()

print("플롯 창을 닫았습니다.")