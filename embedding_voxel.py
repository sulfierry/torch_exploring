import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.colors
import matplotlib.patches as patches

class EmbeddingVoxel:

    def __init__(self, voxel, voxel_grid, coord, center, grid_size, file_path):
        self.voxel = voxel
        self.voxel_grid = voxel_grid
        self.coord = coord
        self.center = center
        self.grid_size = grid_size
        self.parsed_pdb_data = self.parse_pdb(file_path)
        self.property_grid = None
        self.aa_grid = None

    def plot_voxel(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        where_result = np.where(self.voxel == 1)
        
        if len(where_result) == 3:  # Verifica se há 3 arrays para x, y e z.
            x, y, z = where_result
            ax.scatter(x, y, z, alpha=0.6, s=15, c='blue')
        else:
            print("No voxels with value 1 found or voxel array is not 3D.")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Voxel grid')
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
                pass

        return parsed_data
    
    def coord_to_voxel(self, coord):
        voxel_coord = np.floor((coord - self.center) / self.grid_size).astype(int)
        if np.any(voxel_coord < 0) or np.any(voxel_coord >= self.voxel_grid.shape):
            print(f"Invalid voxel coordinates: {voxel_coord} for atom coordinate: {coord}")
            return None
        return voxel_coord
    
    def pdb_to_voxel_atom(self):
        atom_info_grid = np.empty(self.voxel_grid.shape, dtype=object)
        
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in self.parsed_pdb_data[atom_section]:  # Aqui foi feita a correção
                voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is not None:
                    self.voxel_grid[tuple(voxel_coord)] = 1
                    atom_info_grid[tuple(voxel_coord)] = atom_details

        self.voxel = self.voxel_grid  # Update the voxel attribute
        return self.voxel_grid, atom_info_grid


    def get_amino_acid_property(self, res_name):
    
        PROPERTIES = {
            'HIDROPHOBIC': ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL'],
            'POSITIVE': ['ARG', 'HIS', 'LYS'],
            'NEUTRAL': ['ASN', 'CYS', 'GLN', 'GLY', 'SER', 'THR', 'TYR'],
            'NEGATIVE': ['ASP', 'GLU']
        }

        for property, amino_acids in PROPERTIES.items():
            if res_name in amino_acids:
                return property
        return None

    def pdb_to_voxel_property(self):
        self.property_grid = np.empty(self.voxel_grid.shape, dtype=object)

        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in self.parsed_pdb_data[atom_section]:
                voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is not None:
                    aa_property = self.get_amino_acid_property(atom_details['res_name'])
                    if aa_property:
                        self.property_grid[tuple(voxel_coord)] = aa_property

        #return property_grid



    def pdb_to_voxel_amino_acid(self):
        self.aa_grid = np.empty(self.voxel_grid.shape, dtype=object)
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in self.parsed_pdb_data[atom_section]:
                voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is not None:
                    self.aa_grid[tuple(voxel_coord)] = atom_details['res_name']


        #return aa_grid

    def project_sum_with_property_and_aa(self, axis=2):
        projection_sum = np.sum(self.voxel_grid, axis=axis)
        property_projection = np.empty(projection_sum.shape, dtype=object)
        aa_projection = np.empty(projection_sum.shape, dtype=object)
        for x in range(projection_sum.shape[0]):
            for y in range(projection_sum.shape[1]):
                if axis == 0:
                    properties_in_column = self.property_grid[x, y, :]
                    aas_in_column = self.aa_grid[x, y, :]
                elif axis == 1:
                    properties_in_column = self.property_grid[x, :, y]
                    aas_in_column = self.aa_grid[x, :, y]
                else:
                    properties_in_column = self.property_grid[:, x, y]
                    aas_in_column = self.aa_grid[:, x, y]

                properties_in_column = [prop for prop in properties_in_column if prop]
                aas_in_column = [aa for aa in aas_in_column if aa]

                if properties_in_column:
                    unique, counts = np.unique(properties_in_column, return_counts=True)
                    predominant_property = unique[np.argmax(counts)]
                    property_projection[x, y] = predominant_property

                if aas_in_column:
                    unique, counts = np.unique(aas_in_column, return_counts=True)
                    predominant_aa = unique[np.argmax(counts)]
                    aa_projection[x, y] = predominant_aa

        return projection_sum, property_projection, aa_projection



    @staticmethod
    def pdb_to_voxel_atom_with_limited_warnings(voxel_instance):
        
        parsed_pdb = voxel_instance.parsed_pdb_data
        global warnings_count
        warnings_limit = 5
        warnings_count = 0
        atom_info_grid = np.empty(voxel_instance.voxel_grid.shape, dtype=object)
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in parsed_pdb[atom_section]:
                voxel_coord = voxel_instance.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is None:
                    if warnings_count < warnings_limit:
                        print(f"Invalid voxel coordinates for atom coordinate: {atom_details['coord']}")
                        warnings_count += 1
                else:
                    voxel_instance.voxel_grid[tuple(voxel_coord)] = 1
                    atom_info_grid[tuple(voxel_coord)] = atom_details
        if warnings_count == warnings_limit:
            print("... More invalid voxel coordinates found. Limiting output ...")
        return voxel_instance.voxel_grid, atom_info_grid

    @staticmethod
    def plot_projection_with_corrected_representation(projection_sum, aa_projection, property_projection, atom_info_grid, title="Projection"):

        COLOR_PALETTE = {
            'HIDROPHOBIC': "#FFFF00",
            'POSITIVE': "#0000FF",
            'NEUTRAL': "#808080",
            'NEGATIVE': "#FF0000"
        }

        fig, ax = plt.subplots(figsize=(8, 8))
        colored_projection = np.zeros(projection_sum.shape + (3,)) + 1  # initialize with white color

        for x in range(projection_sum.shape[0]):
            for y in range(projection_sum.shape[1]):
                atom_count = projection_sum[x, y]
                if aa_projection[x, y] and atom_count > 0:
                    color = matplotlib.colors.hex2color(COLOR_PALETTE[property_projection[x, y]])
                    min_intensity = 0.3
                    range_intensity = 0.7  # 1 - min_intensity
                    intensity = min_intensity + range_intensity * (atom_count / (projection_sum.max() + 0.5))
                    colored_projection[x, y] = [c * intensity for c in color]

                    # Collecting atom names for the current cell
                    atoms_in_column = atom_info_grid[x, y, :]
                    atom_names = [atom["element"] for atom in atoms_in_column if atom]
                    unique_atoms, atom_counts = np.unique(atom_names, return_counts=True)
                    atom_string = ", ".join([f"{atom}{count}" for atom, count in zip(unique_atoms, atom_counts)])

                    label = f"{aa_projection[x, y]}\n{atom_count}\n{atom_string}"
                    ax.text(y, x, label, ha='center', va='center', color='white', fontsize=6)
                    rect = patches.Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)

        ax.imshow(colored_projection, origin='upper')
        ax.set_title(title)  # Aqui você pode definir o título passando o argumento 'title'
        plt.tight_layout()
        plt.show()

    @staticmethod
    def combine_projections(projection_xz, projection_yz, projection_xy):

        map_color_value = 255
        # Redimensione as projeções para terem a mesma forma
        height, width = projection_xz.shape
        projection_xz_color = np.zeros((height, width, 3), dtype=np.uint8)
        projection_yz_color = np.zeros((height, width, 3), dtype=np.uint8)
        projection_xy_color = np.zeros((height, width, 3), dtype=np.uint8)

        # Aplique mapeamento de cores às projeções
        projection_xz_color[:, :, 0] = (projection_xz * map_color_value).astype(np.uint8)
        projection_yz_color[:, :, 1] = (projection_yz * map_color_value).astype(np.uint8)
        projection_xy_color[:, :, 2] = (projection_xy * map_color_value).astype(np.uint8)

        # Combine as projeções coloridas
        combined_image = projection_xz_color + projection_yz_color + projection_xy_color

        # Certifique-se de que os valores não ultrapassem 255
        combined_image = np.clip(combined_image, 0, map_color_value).astype(np.uint8)

        return combined_image

def read_multiple_pdbs(grid_dim, grid_size, center):
    
   # Loop over all PDB files in the directory
    for pdb_filename in os.listdir("./"):
        if pdb_filename.endswith(".pdb"):
            file_path = os.path.join("./", pdb_filename)
            
            # Create an instance of the EmbeddingVoxel class
            voxel_instance = EmbeddingVoxel(None, np.zeros(grid_dim), None, center, grid_size, file_path)
            
            # Call the methods
            voxel_instance.pdb_to_voxel_atom()
            property_grid = voxel_instance.pdb_to_voxel_property()
            aa_grid = voxel_instance.pdb_to_voxel_amino_acid()
            projection_sum, property_projection, aa_projection = voxel_instance.project_sum_with_property_and_aa()

            # Use the appropriate function for pdb_to_voxel_atom_with_limited_warnings if it's still needed
            voxel_grid, atom_info_grid = EmbeddingVoxel.pdb_to_voxel_atom_with_limited_warnings(voxel_instance)
            
            # Projecting along different axes
            projection_sum_xy, property_projection_xy, aa_projection_xy = voxel_instance.project_sum_with_property_and_aa(axis=2)
            projection_sum_xz, property_projection_xz, aa_projection_xz = voxel_instance.project_sum_with_property_and_aa(axis=0)
            projection_sum_yz, property_projection_yz, aa_projection_yz = voxel_instance.project_sum_with_property_and_aa(axis=1)

            combined_image = EmbeddingVoxel.combine_projections(projection_sum_xz, projection_sum_yz, projection_sum_xy)
            # Exiba a imagem combinada
            plt.imshow(combined_image)
            plt.axis('off')  # Desligue as bordas do eixo
            plt.title('f(x, y, z) = (2x+y, y+2z)')
            plt.show()




if __name__ == "__main__":

    # Initializing and running
    grid_dim = [10, 10, 10] # grid dimension in voxel unitis
    grid_size = 1.0 # Ångström (Å)
    center = np.array([28.891, -0.798, 65.003]) 
    file_path = "./3c9t.pdb"
    
    voxel_instance = EmbeddingVoxel(None, np.zeros(grid_dim), None, center, grid_size, file_path)
    voxel_instance.pdb_to_voxel_atom()

    property_grid = voxel_instance.pdb_to_voxel_property()
    aa_grid = voxel_instance.pdb_to_voxel_amino_acid()
    projection_sum, property_projection, aa_projection = voxel_instance.project_sum_with_property_and_aa()

    # Use the appropriate function for pdb_to_voxel_atom_with_limited_warnings if it's still needed
    voxel_grid, atom_info_grid = EmbeddingVoxel.pdb_to_voxel_atom_with_limited_warnings(voxel_instance)
    
    projection_sum_xy, property_projection_xy, aa_projection_xy = voxel_instance.project_sum_with_property_and_aa(axis=2)  # Para projeção X, Y
    projection_sum_xz, property_projection_xz, aa_projection_xz = voxel_instance.project_sum_with_property_and_aa(axis=1)  # Para projeção X, Z
    projection_sum_yz, property_projection_yz, aa_projection_yz = voxel_instance.project_sum_with_property_and_aa(axis=0)  # Para projeção Y, Z
    
    # voxel_instance.plot_voxel()
    # EmbeddingVoxel.plot_projection_with_corrected_representation(projection_sum_xy, aa_projection_xy, property_projection_xy, atom_info_grid, title="Projection (x,y)")
    # EmbeddingVoxel.plot_projection_with_corrected_representation(projection_sum_xz, aa_projection_xz, property_projection_xz, atom_info_grid, title="Projection (x,z)")
    # EmbeddingVoxel.plot_projection_with_corrected_representation(projection_sum_yz, aa_projection_yz, property_projection_yz, atom_info_grid, title="Projection (y,z)")

    combined_image = EmbeddingVoxel.combine_projections(projection_sum_xz, projection_sum_yz, projection_sum_xy)
    # Exiba a imagem combinada
    # plt.imshow(combined_image)
    # plt.axis('off')  # Desligue as bordas do eixo
    # plt.title('f(x, y, z) = (2x+y, y+2z)')
    # plt.show()

    read_multiple_pdbs(grid_dim, grid_size, center)

