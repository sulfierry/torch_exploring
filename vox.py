import math
import numpy as np
import matplotlib.pyplot as plt

# Função de plot 3D para visualizar os voxels
def plot_voxels(voxel_grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, edgecolor="k")
    plt.show()


def plot_voxel(voxel):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.where(voxel == 1)
    ax.scatter(x, y, z, alpha=0.6, s=15, c='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

import numpy as np

# Função para analisar o arquivo PDB
def parse_pdb(pdb_file_path):
    with open(pdb_file_path, 'r') as file:
        pdb_data = file.readlines()
    
    parsed_data = {
        'chains': [],
        'cofactors': [],
        'ligands': []
    }

    for line in pdb_data:
        if line.startswith("ATOM"):
            parsed_atom = {
                'name': line[12:16].strip(),
                'altLoc': line[16].strip(),
                'res_name': line[17:20].strip(),
                'chain_id': line[21].strip(),
                'res_seq': int(line[22:26].strip()),
                'icode': line[26].strip(),
                'coord': [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                'occupancy': float(line[54:60]),
                'temp_factor': float(line[60:66]),
                'element': line[76:78].strip(),
                'charge': line[78:80].strip()
            }
            parsed_data['chains'].append(parsed_atom)
        elif line.startswith(("HETATM", "ANISOU", "TER", "CONECT")):
            # Processa outras linhas conforme necessário
            pass  # Placeholder para processamento adicional

    return parsed_data

# Função para converter coordenadas do PDB para índices de voxel
def coord_to_voxel(coord, origin, grid_size):
    voxel_coord = np.floor((coord - origin) / grid_size).astype(int)
    if np.any(voxel_coord < 0) or np.any(voxel_coord >= grid_dim):
        print(f"Invalid voxel coordinates: {voxel_coord} for atom coordinate: {coord}")
        return None
    return voxel_coord

# Função para converter o PDB (no formato de dicionário) para um voxel grid
def pdb_to_voxel_atom(parsed_pdb):
    voxel_grid = np.zeros(grid_dim, dtype= int)
    atom_info_grid = np.empty(grid_dim, dtype=object)

    for atom_section in ["chains", "cofactors", "ligands"]:
        for atom_details in parsed_pdb[atom_section]:
            voxel_coord = coord_to_voxel(np.array(atom_details["coord"]), origin, grid_size)
            if voxel_coord is not None:
                voxel_grid[tuple(voxel_coord)] = 1
                atom_info_grid[tuple(voxel_coord)] = atom_details

    return voxel_grid, atom_info_grid

# Função para converter o PDB (no formato de dicionário) para uma voxel grid representando aminoácidos
def pdb_to_voxel_residue(parsed_pdb):
    voxel_grid = np.zeros(grid_dim, dtype=int)
    residue_info_grid = np.empty(grid_dim, dtype=object)

    seen_residues = set()

    for atom_section in ["chains", "cofactors", "ligands"]:
        for atom_details in parsed_pdb[atom_section]:
            residue_identifier = (atom_details['chain_id'], atom_details['res_seq'], atom_details['icode'])
            if residue_identifier not in seen_residues:
                seen_residues.add(residue_identifier)
                if atom_details['name'] == 'CA':
                    voxel_coord = coord_to_voxel(np.array(atom_details["coord"]), origin, grid_size)
                    if voxel_coord is not None:
                        voxel_grid[tuple(voxel_coord)] = 1
                        residue_info = {
                            'res_name': atom_details['res_name'],
                            'chain_id': atom_details['chain_id'],
                            'res_seq': atom_details['res_seq'],
                            'icode': atom_details['icode']
                        }
                        residue_info_grid[tuple(voxel_coord)] = residue_info

    return voxel_grid, residue_info_grid


def project_maximal(voxel_grid, axis=2):
    """
        Esta abordagem seleciona o valor máximo ao longo de uma dimensão. 
        É útil quando a presença de um recurso (neste caso, um átomo ou aminoácido) 
        em qualquer posição ao longo de uma dimensão é considerada importante.
    """
    return np.max(voxel_grid, axis=axis)

def project_sum(voxel_grid, axis=2):
    """
        Projeção de Soma:
        Esta abordagem soma os valores ao longo de uma dimensão. 
        É útil quando você está interessado em saber quantos átomos ou aminoácidos existem ao longo dessa dimensão.
    """
    return np.sum(voxel_grid, axis=axis)


def radial_projection(voxel_grid, center):
    # As dimensões do voxel grid
    depth, height, width = voxel_grid.shape

    # Crie uma imagem 2D vazia com a mesma largura e altura do voxel grid
    projected_image = np.zeros((height, width))

    # Para cada voxel, calcule sua distância e ângulo ao centro
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if voxel_grid[z, y, x]:  # Se houver um átomo/aminoácido neste voxel
                    # Calcule a distância ao centro
                    dist = np.linalg.norm(np.array([z, y, x]) - center)
                    
                    # Calcule o ângulo em relação ao plano XY
                    # Por simplicidade, vamos usar apenas o ângulo em relação ao eixo Y (pode ser ajustado conforme necessário)
                    angle = np.arctan2(x - center[2], y - center[1])

                    # Converta o ângulo para um índice de pixel
                    # Usaremos o ângulo para determinar a posição x e a distância para determinar a posição y
                    projected_x = int((angle + np.pi) / (2 * np.pi) * width)
                    projected_y = int(dist / np.linalg.norm([height, width]) * height)

                    # Defina o valor no pixel correspondente
                    projected_image[projected_y, projected_x] = 1

    return projected_image

# Definição do tamanho e origem da grade
grid_dim = [25, 25, 25]
grid_size = 1.0
origin = np.array([17.773, 63.285, 121.743])

parsed_pdb = parse_pdb("./3c9t.pdb")  
atom_voxel_grid, atom_info_grid = pdb_to_voxel_atom(parsed_pdb)
aac_voxel_grid, acc_info_grid = pdb_to_voxel_residue(parsed_pdb)

print(atom_info_grid.shape)
#plot_voxel(atom_voxel_grid)
# plot_voxel(aac_voxel_grid)
"""armazenar informações adicionais sobre cada átomo quando estiver preenchendo sua grade de voxels."""

# atom_vector = atom_voxel_grid.flatten()
# aac_vector = aac_voxel_grid.flatten()
# print(atom_vector.shape, aac_vector.shape)

# Aplicar a função
projected_atom_grid = project_maximal(atom_voxel_grid)
projected_acc_grid = project_maximal(aac_voxel_grid)

# Visualizar usando matplotlib
plt.imshow(projected_atom_grid, cmap='gray')
plt.title('Projected Atom Grid')
plt.show()

plt.imshow(projected_acc_grid, cmap='gray')
plt.title('Projected Amino Acid Grid')
plt.show()


# Aplicar a função
projected_atom_grid_sum = project_sum(atom_voxel_grid)
projected_acc_grid_sum = project_sum(aac_voxel_grid)

# Visualizar usando matplotlib
plt.imshow(projected_atom_grid_sum, cmap='gray')
plt.title('Sum Projected Atom Grid')
plt.show()

plt.imshow(projected_acc_grid_sum, cmap='gray')
plt.title('Sum Projected Amino Acid Grid')
plt.show()