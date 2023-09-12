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
    # OK
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
    # OK
    def coord_to_voxel(self, coord):
        voxel_coord = np.floor((coord - self.center) / self.grid_size).astype(int)
        if np.any(voxel_coord < 0) or np.any(voxel_coord >= self.voxel_grid.shape):
            print(f"Invalid voxel coordinates: {voxel_coord} for atom coordinate: {coord}")
            return None
        return voxel_coord
    # OK
    def pdb_to_voxel_atom(self, parsed_pdb):
        atom_info_grid = np.empty(self.voxel_grid.shape, dtype=object)
        
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in parsed_pdb[atom_section]:
                voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is not None:
                    self.voxel_grid[tuple(voxel_coord)] = 1
                    atom_info_grid[tuple(voxel_coord)] = atom_details

        return self.voxel_grid, atom_info_grid

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

# OK
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
# OK
def pdb_to_voxel_amino_acid(parsed_pdb, voxel_instance):
    aa_grid = np.empty(voxel_instance.voxel_grid.shape, dtype=object)
    for atom_section in ["chains", "cofactors", "ligands"]:
        for atom_details in parsed_pdb[atom_section]:
            voxel_coord = voxel_instance.coord_to_voxel(np.array(atom_details["coord"]))
            if voxel_coord is not None:
                aa_grid[tuple(voxel_coord)] = atom_details['res_name']
    return aa_grid

# OK
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


def plot_projection_with_corrected_labels(projection_sum, aa_projection, property_projection):
    """Plot the projection with amino acid property labels, borders, and amino acid names,
    ensuring even low intensity squares are visible."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colored_projection = np.zeros(projection_sum.shape + (3,))
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if property_projection[x, y]:
                color = matplotlib.colors.hex2color(PDBVoxel.COLOR_PALETTE[property_projection[x, y]])
                intensity = (projection_sum[x, y] + 0.5) / (projection_sum.max() + 0.5)  # Adjusted intensity
                colored_projection[x, y] = [c * intensity for c in color]
    
    ax.imshow(colored_projection)
    ax.set_title('Sum Projection with Amino Acid Properties')
    
    # Add property labels and borders to each colored square
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if aa_projection[x, y]:
                ax.text(y, x, aa_projection[x, y], ha='center', va='center', color='white', fontsize=10)
                # Draw a white border around the square
                rect = patches.Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
    
    patches_list = [patches.Patch(color=PDBVoxel.COLOR_PALETTE[prop], label=prop) for prop in PDBVoxel.COLOR_PALETTE]
    ax.legend(handles=patches_list, loc='center left', bbox_to_anchor=(1, 0.5))
    print(f"Projection_sum: {type(projection_sum)} | AA projection: {type(aa_projection)} | Property projection: {type(property_projection)}")
    plt.tight_layout()
    plt.show()


# OK
def plot_projection_by_property_with_labels_adjusted(projection_sum, aa_projection, target_property):
    """Plot the projection filtered by a specific amino acid property with amino acid labels and borders, 
    ensuring even low intensity squares are visible."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Filter the projection to only the target property
    filtered_projection = np.zeros(projection_sum.shape + (3,))
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if property_projection[x, y] == target_property:
                color = matplotlib.colors.hex2color(PDBVoxel.COLOR_PALETTE[target_property])
                intensity = (projection_sum[x, y] + 0.5) / (projection_sum.max() + 0.5)  # Adjusted intensity
                filtered_projection[x, y] = [c * intensity for c in color]
    
    ax.imshow(filtered_projection)
    ax.set_title(f'Sum Projection of {target_property} Amino Acids')
    
    # Add labels to each colored square with white font
    for x in range(projection_sum.shape[0]):
        for y in range(projection_sum.shape[1]):
            if aa_projection[x, y] and property_projection[x, y] == target_property:
                ax.text(y, x, aa_projection[x, y], ha='center', va='center', color='white')
                # Draw a white border around the square
                rect = patches.Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    # Initializing and running
    grid_dim = [15, 15, 15]
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

    # Generate the plots for each amino acid property with adjusted visibility and borders
    for property_type in PDBVoxel.PROPERTIES:
        plot_projection_by_property_with_labels_adjusted(projection_sum, aa_projection, property_type)