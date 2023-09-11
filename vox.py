import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.colors
import matplotlib.patches as patches

class PDBVoxel:
    PROPERTIES = {
        'HIDROPHOBIC': ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL'],
        'POSITIVE': ['ARG', 'HIS', 'LYS'],
        'NEUTRAL': ['ASN', 'CYS', 'GLN', 'GLY', 'SER', 'THR', 'TYR'],
        'NEGATIVE': ['ASP', 'GLU']
    }

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
                pass

        return parsed_data
    
    def coord_to_voxel(self, coord):
        voxel_coord = np.floor((coord - self.center) / self.grid_size).astype(int)
        if np.any(voxel_coord < 0) or np.any(voxel_coord >= self.voxel_grid.shape):
            print(f"Invalid voxel coordinates: {voxel_coord} for atom coordinate: {coord}")
            return None
        return voxel_coord

    def pdb_to_voxel_atom(self, parsed_pdb):
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

    def project_sum(self, axis=2):
        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis. Choose from 0, 1, or 2.")
        return np.sum(self.voxel_grid, axis=axis)


# Functions outside the class
def get_amino_acid_property(res_name):
    for property, amino_acids in PDBVoxel.PROPERTIES.items():
        if res_name in amino_acids:
            return property
    return None


def pdb_to_voxel_property(parsed_pdb, voxel_instance):
    property_grid = np.empty(voxel_instance.voxel_grid.shape, dtype=object)
    for atom_section in ["chains", "cofactors", "ligands"]:
        for atom_details in parsed_pdb[atom_section]:
            voxel_coord = voxel_instance.coord_to_voxel(np.array(atom_details["coord"]))
            if voxel_coord is not None:
                aa_property = get_amino_acid_property(atom_details['res_name'])
                if aa_property:
                    property_grid[tuple(voxel_coord)] = aa_property
    return property_grid

def pdb_to_voxel_amino_acid(parsed_pdb, voxel_instance):
    aa_grid = np.empty(voxel_instance.voxel_grid.shape, dtype=object)
    for atom_section in ["chains", "cofactors", "ligands"]:
        for atom_details in parsed_pdb[atom_section]:
            voxel_coord = voxel_instance.coord_to_voxel(np.array(atom_details["coord"]))
            if voxel_coord is not None:
                aa_grid[tuple(voxel_coord)] = atom_details['res_name']
    return aa_grid

def project_sum_with_property_and_aa(voxel_instance, property_grid, aa_grid, axis=2):
    projection_sum = np.sum(voxel_instance.voxel_grid, axis=axis)
    property_projection = np.empty(projection_sum.shape, dtype=object)
    aa_projection = np.empty(projection_sum.shape, dtype=object)
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if axis == 0:
                properties_in_column = property_grid[x, y, :]
                aas_in_column = aa_grid[x, y, :]
            elif axis == 1:
                properties_in_column = property_grid[x, :, y]
                aas_in_column = aa_grid[x, :, y]
            else:
                properties_in_column = property_grid[:, x, y]
                aas_in_column = aa_grid[:, x, y]
            
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


def project_sum_with_property(voxel_instance, property_grid, axis=2):
    projection_sum = np.sum(voxel_instance.voxel_grid, axis=axis)
    property_projection = np.empty(projection_sum.shape, dtype=object)
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if axis == 0:
                properties_in_column = property_grid[x, y, :]
            elif axis == 1:
                properties_in_column = property_grid[x, :, y]
            else:
                properties_in_column = property_grid[:, x, y]
            properties_in_column = [prop for prop in properties_in_column if prop]
            if properties_in_column:
                unique, counts = np.unique(properties_in_column, return_counts=True)
                predominant_property = unique[np.argmax(counts)]
                property_projection[x, y] = predominant_property
    return projection_sum, property_projection


def plot_projection_with_property(projection_sum, property_projection):
    fig, ax = plt.subplots(figsize=(10, 10))
    colored_projection = np.zeros(projection_sum.shape + (3,))
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if property_projection[x, y]:
                color = matplotlib.colors.hex2color(PDBVoxel.COLOR_PALETTE[property_projection[x, y]])
                intensity = projection_sum[x, y] / projection_sum.max()
                colored_projection[x, y] = [c * intensity for c in color]
    ax.imshow(colored_projection)
    ax.set_title('Sum Projection with Amino Acid Properties')
    patches_list = [patches.Patch(color=PDBVoxel.COLOR_PALETTE[prop], label=prop) for prop in PDBVoxel.COLOR_PALETTE]
    
    # Place the legend outside the plot
    ax.legend(handles=patches_list, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_projection_with_corrected_labels(projection_sum, aa_projection, property_projection):
    """Plot the projection with white amino acid names and borders around the perimeter of non-empty voxels."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use the palette for amino acid properties
    colored_projection = np.zeros(projection_sum.shape + (3,))
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if property_projection[x, y]:
                color = matplotlib.colors.hex2color(PDBVoxel.COLOR_PALETTE[property_projection[x, y]])
                intensity = projection_sum[x, y] / projection_sum.max()
                colored_projection[x, y] = [c * intensity for c in color]
    
    ax.imshow(colored_projection)
    ax.set_title('Sum Projection with Specific Amino Acids')
    
    # Add labels (amino acid names) to each colored square with white font
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if aa_projection[x, y]:
                ax.text(y, x, aa_projection[x, y], ha='center', va='center', color='white')
                # Draw a white border around the square
                rect = patches.Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
    
    patches_list = [patches.Patch(color=PDBVoxel.COLOR_PALETTE[prop], label=prop) for prop in PDBVoxel.COLOR_PALETTE]
    ax.legend(handles=patches_list, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def project_sum_with_amino_acid(voxel_instance, aa_grid, axis=2):
    projection_sum = np.sum(voxel_instance.voxel_grid, axis=axis)
    aa_projection = np.empty(projection_sum.shape, dtype=object)
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if axis == 0:
                aas_in_column = aa_grid[x, y, :]
            elif axis == 1:
                aas_in_column = aa_grid[x, :, y]
            else:
                aas_in_column = aa_grid[:, x, y]
            aas_in_column = [aa for aa in aas_in_column if aa]
            if aas_in_column:
                unique, counts = np.unique(aas_in_column, return_counts=True)
                predominant_aa = unique[np.argmax(counts)]
                aa_projection[x, y] = predominant_aa
    return projection_sum, aa_projection


# Initializing and running
grid_dim = [10, 10, 10]
grid_size = 1.0
center = np.array([17.773, 63.285, 121.743])
file_path = "./3c9t.pdb"

voxel_instance = PDBVoxel(None, np.zeros(grid_dim), None, center, grid_size)
parsed_pdb = voxel_instance.parse_pdb(file_path)
voxel_instance.pdb_to_voxel_atom(parsed_pdb)
property_grid = pdb_to_voxel_property(parsed_pdb, voxel_instance)
aa_grid = pdb_to_voxel_amino_acid(parsed_pdb, voxel_instance)
projection_sum, property_projection, aa_projection = project_sum_with_property_and_aa(voxel_instance, property_grid, aa_grid)
plot_projection_with_corrected_labels(projection_sum, aa_projection, property_projection)
