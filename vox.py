
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PDBVoxel:

        # Classificação dos aminoácidos
    PROPERTIES = {
        'HIDROPHOBIC': ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL'],
        'POSITIVE': ['ARG', 'HIS', 'LYS'],
        'NEUTRAL': ['ASN', 'CYS', 'GLN', 'GLY', 'SER', 'THR', 'TYR'],
        'NEGATIVE': ['ASP', 'GLU']
    }

    # Paleta de cores com base nas propriedades
    COLOR_PALETTE = {
        'HIDROPHOBIC': "#FFFF00",
        'POSITIVE': "#0000FF",
        'NEUTRAL': "#808080",
        'NEGATIVE': "#FF0000"
    }

    def __init__(self, voxel, voxel_grid, coord, center, grid_size):
        self.voxel = voxel
        self.voxel_grid = voxel_grid
        self.coord = coord
        self.center = center
        self.grid_size = grid_size

    def plot_voxels(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(self.voxel_grid, edgecolor="k")
        plt.show()

    def plot_voxel(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = np.where(self.voxel == 1)
        ax.scatter(x, y, z, alpha=0.6, s=15, c='blue')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def parse_pdb(self, pdb_file_path):
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
    
    def coord_to_voxel(self, coord):
        voxel_coord = np.floor((coord - self.center) / self.grid_size).astype(int)
        if np.any(voxel_coord < 0) or np.any(voxel_coord >= self.voxel_grid.shape):
            print(f"Invalid voxel coordinates: {voxel_coord} for atom coordinate: {self.coord}")
            return None
        return voxel_coord


    def pdb_to_voxel_atom(self, parsed_pdb):
        global atom_info_grid
        atom_info_grid = np.empty(self.voxel_grid.shape, dtype=object)
        
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in parsed_pdb[atom_section]:
                voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is not None:
                    self.voxel_grid[tuple(voxel_coord)] = 1
                    atom_info_grid[tuple(voxel_coord)] = atom_details

        return self.voxel_grid, atom_info_grid

    def pdb_to_voxel_residue(self, parsed_pdb):
        residue_info_grid = np.empty(self.voxel_grid.shape, dtype=object)

        seen_residues = set()

        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in parsed_pdb[atom_section]:
                residue_identifier = (atom_details['chain_id'], atom_details['res_seq'], atom_details['icode'])
                if residue_identifier not in seen_residues:
                    seen_residues.add(residue_identifier)
                    if atom_details['name'] == 'CA':
                        voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                        if voxel_coord is not None:
                            self.voxel_grid[tuple(voxel_coord)] = 1
                            residue_info = {
                                'res_name': atom_details['res_name'],
                                'chain_id': atom_details['chain_id'],
                                'res_seq': atom_details['res_seq'],
                                'icode': atom_details['icode']
                            }
                            residue_info_grid[tuple(voxel_coord)] = residue_info

        return self.voxel_grid, residue_info_grid

    def project_maximal(self, axis=2):
        """
            Esta abordagem seleciona o valor máximo ao longo de uma dimensão. 
            É útil quando a presença de um recurso (neste caso, um átomo ou aminoácido) 
            em qualquer posição ao longo de uma dimensão é considerada importante.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis. Choose from 0, 1, or 2.")
        return np.max(self.voxel_grid, axis=axis)
    
    def project_sum(self, axis=2):
        """
            Projeção de Soma:
            Esta abordagem soma os valores ao longo de uma dimensão. 
            É útil quando você está interessado em saber quantos átomos ou aminoácidos existem ao longo dessa dimensão.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis. Choose from 0, 1, or 2.")
        return np.sum(self.voxel_grid, axis=axis)

    def radial_projection(self):
        """
            A transformação radial, ou projeção radial, é uma técnica que representa pontos 3D em um plano 2D, 
            usando o raio e o ângulo em relação a um ponto central ou de origem. Esta abordagem pode ser particularmente 
            útil para visualizar estruturas 3D que são centralmente simétricas ou que podem ser bem representadas em termos 
            de distâncias e ângulos a partir de um ponto central.
        """
        # Pegar o ponto central do voxel grid
        center_voxel = np.array(self.voxel_grid.shape) // 2

        # Calcular distâncias de todos os pontos ao centro
        y, x, z = np.ogrid[:self.voxel_grid.shape[0], :self.voxel_grid.shape[1], :self.voxel_grid.shape[2]]
        distances = np.sqrt((y - center_voxel[0])**2 + (x - center_voxel[1])**2 + (z - center_voxel[2])**2)

        # Obter índice da distância máxima (ou raio máximo)
        max_distance = int(np.max(distances))

        # Inicializar projeção radial como zeros
        radial_proj = np.zeros((max_distance,))

        for r in range(max_distance):
            radial_proj[r] = np.sum(self.voxel_grid[distances <= r])

        return radial_proj



# Definição do tamanho e origem da grade
grid_dim = [10, 10, 10]
grid_size = 1.0
center = np.array([17.773, 63.285, 121.743])
file_path = "./3c9t.pdb"


# Criação de uma instância PDBVoxel
voxel_instance = PDBVoxel(None, np.zeros(grid_dim), None, center, grid_size)
parsed_pdb = voxel_instance.parse_pdb(file_path)
voxel_instance.pdb_to_voxel_atom(parsed_pdb)
projection_sum = voxel_instance.project_sum()

plt.figure(figsize=(6, 6))
plt.title('Sum Projection')
plt.imshow(projection_sum, cmap='viridis')
plt.colorbar(label='Total Atoms')
plt.show()


