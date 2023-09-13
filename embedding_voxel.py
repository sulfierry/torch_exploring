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
        """Plota o voxel em uma grid 3D."""
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
        """Converte o PDB para a estrutura de dados dicionario."""
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
        """
        Converta coordenadas atômicas 3D do PDB em coordenadas de voxel na grade.
        
        Parameters:
        - coord: Coordenadas 3D do átomo (do PDB).

        Returns:
        - voxel_coord: Coordenadas correspondentes na grade de voxel ou None se estiver fora da grade.
        """
        
        voxel_coord = np.floor((coord - self.center) / self.grid_size).astype(int)
        
        # Corrigindo a validação
        if np.any(voxel_coord < 0) or np.any(voxel_coord >= np.array(self.voxel_grid.shape)):
            print(f"Invalid voxel coordinates: {voxel_coord} for atom coordinate: {coord}")
            return None
        return voxel_coord

    def pdb_to_voxel_atom(self):
        """
        Mapeia átomos do PDB na grade de voxel.
        Este método processa diferentes seções do PDB: cadeias, cofatores e ligantes.
        
        Returns:
        - self.voxel_grid: Uma grade 3D onde 1 indica a presença de um átomo e 0 indica vazio.
        - atom_info_grid: Uma grade 3D armazenando os detalhes do átomo correspondente ao voxel.
        """
        
        # Inicializando com dtype=object para flexibilidade, mas pode ser menos eficiente
        atom_info_grid = np.empty(self.voxel_grid.shape, dtype=object)
        
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in self.parsed_pdb_data[atom_section]:
                voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                
                if voxel_coord is not None:
                    # Registro de ocupação
                    if self.voxel_grid[tuple(voxel_coord)] == 1:
                        print(f"Warning: Voxel at {voxel_coord} is already occupied. Overwriting!")
                    
                    self.voxel_grid[tuple(voxel_coord)] = 1
                    atom_info_grid[tuple(voxel_coord)] = atom_details

        # Eliminando redundância de variáveis
        return self.voxel_grid, atom_info_grid

    def get_amino_acid_property(self, res_name):
        """
        Obtém a propriedade de um aminoácido com base no seu nome.
        """
        
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

    def map_amino_acid_properties_to_voxel(self):
        """
        Mapeia as propriedades dos aminoácidos para a grade de voxel.
        """
        
        # Inicialize a grade de propriedades
        self.property_grid = np.empty(self.voxel_grid.shape, dtype=object)

        # Vamos processar apenas as chains para esta funcionalidade, já que ligantes e cofatores 
        # podem não ter res_names que correspondam aos aminoácidos padrão.
        for atom_details in self.parsed_pdb_data["chains"]:
            
            voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
            if voxel_coord is not None:
                
                # Verifique se o voxel já está ocupado.
                if self.property_grid[tuple(voxel_coord)]:
                    print(f"Warning: Voxel at {voxel_coord} is already occupied. Overwriting!")
                
                aa_property = self.get_amino_acid_property(atom_details.get('res_name', ''))
                if aa_property:
                    self.property_grid[tuple(voxel_coord)] = aa_property
                else:
                    print(f"Warning: Amino acid {atom_details.get('res_name', '')} not recognized or it's a non-standard amino acid.")

    def pdb_to_voxel_amino_acid(self):
        """
        Converte a informação PDB para uma grade de voxel que representa o nome do aminoácido.
        """
        # Inicializando a grid de aminoácidos
        self.aa_grid = np.empty(self.voxel_grid.shape, dtype=object)

        # Iterando sobre as seções do PDB
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in self.parsed_pdb_data[atom_section]:
                voxel_coord = self.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is not None:
                    # Verifica se o voxel já está ocupado
                    if self.aa_grid[tuple(voxel_coord)]:
                        print(f"Warning: Voxel at {voxel_coord} is already occupied. Overwriting!")
                    self.aa_grid[tuple(voxel_coord)] = atom_details['res_name']
        return self.aa_grid

    def project_sum_with_property_and_aa(self, axis=2):
        # Calculate the sum of the voxel grid along the specified axis.
        projection_sum = np.sum(self.voxel_grid, axis=axis)

        # Create empty dictionaries to store the projected property and AA values.
        property_projection = {}
        aa_projection = {}

        # Iterate over each row and column in the projection.
        for x in range(projection_sum.shape[0]):
            for y in range(projection_sum.shape[1]):

                # Check the corresponding column in the voxel grid, property grid, and AA grid.
                if axis == 0:
                    properties_in_column = self.property_grid[x, y, :]
                    aas_in_column = self.aa_grid[x, y, :]
                elif axis == 1:
                    properties_in_column = self.property_grid[x, :, y]
                    aas_in_column = self.aa_grid[x, :, y]
                else:
                    properties_in_column = self.property_grid[:, x, y]
                    aas_in_column = self.aa_grid[:, x, y]

                # If the column is not empty, calculate the predominant property and AA in the column and store them in the corresponding dictionaries.
                properties_in_column = [prop for prop in properties_in_column if prop]
                aas_in_column = [aa for aa in aas_in_column if aa]

                if properties_in_column:
                    predominant_property = properties_in_column[0]
                    property_projection[x, y] = predominant_property

                if aas_in_column:
                    predominant_aa = aas_in_column[0]
                    aa_projection[x, y] = predominant_aa

        # Return the projection sum, projected property, and projected AA arrays.
        return projection_sum, property_projection, aa_projection

    @staticmethod
    def pdb_to_voxel_atom_with_limited_warnings(voxel_instance):

        parsed_pdb = voxel_instance.parsed_pdb_data
        global warnings_count
        warnings_count = 0
        atom_info_grid = np.empty(voxel_instance.voxel_grid.shape, dtype=object)
        for atom_section in ["chains", "cofactors", "ligands"]:
            for atom_details in parsed_pdb[atom_section]:
                voxel_coord = voxel_instance.coord_to_voxel(np.array(atom_details["coord"]))
                if voxel_coord is not None:
                    voxel_instance.voxel_grid[tuple(voxel_coord)] = 1
                    atom_info_grid[tuple(voxel_coord)] = atom_details
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
        colored_projection = np.zeros(projection_sum.shape + (3,))

        for x in range(projection_sum.shape[0]):
            for y in range(projection_sum.shape[1]):
                atom_count = projection_sum[x, y]
                if aa_projection[x, y] and atom_count > 0:
                    color = matplotlib.colors.hex2color(COLOR_PALETTE[property_projection[x, y]])
                    colored_projection[x, y] = color

                    # Collecting atom names for the current cell
                    atoms_in_column = atom_info_grid[x, y, :]
                    atom_names = [atom["element"] for atom in atoms_in_column if atom]

                    label = f"{aa_projection[x, y]}\n{atom_count}\n{atom_names}"
                    ax.text(y, x, label, ha='center', va='center', color='white', fontsize=6)
                    rect = patches.Rectangle((y-0.5, x-0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
                    ax.add_patch(rect)

        ax.imshow(colored_projection, origin='upper')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def combine_projections(projection_xz, projection_yz, projection_xy):
        """
        Combina três projeções em um único mapa de cores RGB e normaliza o resultado.
        
        :param projection_xz: Array 2D representando a projeção XZ.
        :param projection_yz: Array 2D representando a projeção YZ.
        :param projection_xy: Array 2D representando a projeção XY.
        :return: Imagem RGB combinada e normalizada.
        """
        
        # Validação de entrada: Garantir que as projeções tenham as mesmas dimensões
        assert projection_xz.shape == projection_yz.shape == projection_xy.shape, "Projections must have the same shape"
        
        map_color_value = 255
        height, width = projection_xz.shape
        
        # Inicializar imagens coloridas com zeros
        projection_xz_color = np.zeros((height, width, 3), dtype=np.float32)  # Usar float32 para evitar overflow antes da normalização
        projection_yz_color = np.zeros((height, width, 3), dtype=np.float32)
        projection_xy_color = np.zeros((height, width, 3), dtype=np.float32)

        # Mapear projeções para seus respectivos canais de cores
        projection_xz_color[:, :, 0] = projection_xz * map_color_value
        projection_yz_color[:, :, 1] = projection_yz * map_color_value
        projection_xy_color[:, :, 2] = projection_xy * map_color_value

        # Combinação de imagens coloridas
        combined_image = projection_xz_color + projection_yz_color + projection_xy_color

        # Normalização usando Min-Max scaling
        min_pixel = combined_image.min()
        max_pixel = combined_image.max()
        normalized_image = ((combined_image - min_pixel) / (max_pixel - min_pixel) * map_color_value).astype(np.uint8)

        return normalized_image

    @staticmethod
    def read_multiple_pdbs(grid_dim, grid_size, center):
        
    # Loop over all PDB files in the directory
        for pdb_filename in os.listdir("./"):
            if pdb_filename.endswith(".pdb"):
                file_path = os.path.join("./", pdb_filename)
                
                # Create an instance of the EmbeddingVoxel class
                voxel_instance = EmbeddingVoxel(None, np.zeros(grid_dim), None, center, grid_size, file_path)
                
                # Call the methods
                voxel_instance.pdb_to_voxel_atom()
                property_grid = voxel_instance.map_amino_acid_properties_to_voxel()
                aa_grid = voxel_instance.pdb_to_voxel_amino_acid()

                # Projecting along different axes
                projection_sum_xy, property_projection_xy, aa_projection_xy = voxel_instance.project_sum_with_property_and_aa(axis=2)
                projection_sum_xz, property_projection_xz, aa_projection_xz = voxel_instance.project_sum_with_property_and_aa(axis=0)
                projection_sum_yz, property_projection_yz, aa_projection_yz = voxel_instance.project_sum_with_property_and_aa(axis=1)

                combined_image = EmbeddingVoxel.combine_projections(projection_sum_xz, projection_sum_yz, projection_sum_xy)
                # Exiba a imagem combinada
                save_path = f"./{pdb_filename}_combined_projection.png"
                plt.imshow(combined_image)
                plt.axis('off')  # Turn off axis borders
                # plt.title('f(x, y, z) = (2x+y, y+2z)')
                plt.savefig(save_path, dpi=600)
                plt.show()
                plt.close()



if __name__ == "__main__":

    # Initializing and running
    grid_dim = [10, 10, 10] # grid dimension in voxel unitis
    grid_size = 1.0 # Ångström (Å)
    center = np.array([28.891, -0.798, 65.003]) 
    file_path = "./3c9t.pdb"
    
    voxel_instance = EmbeddingVoxel(None, np.zeros(grid_dim), None, center, grid_size, file_path)
    voxel_instance.pdb_to_voxel_atom()

    property_grid = voxel_instance.map_amino_acid_properties_to_voxel()
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
    plt.imshow(combined_image)
    plt.axis('off')  # Desligue as bordas do eixo
    plt.title('f(x, y, z) = (2x+y, y+2z)')
    plt.show()

    # EmbeddingVoxel.read_multiple_pdbs(grid_dim, grid_size, center)

"""
    ResNet-18: Uma das variantes mais simples da família ResNet, com 18 camadas.
    VGG-16: Uma das redes VGG mais populares, com 16 camadas ponderadas.
    ResNet-34: Uma variante da ResNet com 34 camadas.
    VGG-19: Uma variante mais profunda da VGG, com 19 camadas ponderadas.
    ResNet-50: Uma ResNet mais profunda com blocos de "bottle-neck", totalizando 50 camadas.
    DenseNet-121: O número indica a profundidade (neste caso, 121 camadas). A DenseNet tem conexões densas entre as camadas.
    ResNet-101: Uma variante ainda mais profunda da ResNet, com 101 camadas.
    ResNet-152: A variante mais profunda comumente usada da ResNet, com 152 camadas.
    DenseNet-169, DenseNet-201, DenseNet-264: Variantes mais profundas da DenseNet.
    EfficientNet: Esta arquitetura tem várias variantes (B0 a B7) que são escalonadas em profundidade, largura e resolução. A EfficientNet-B0 é a mais leve, enquanto a B7 é a mais pesada e complexa.
    Outras arquiteturas como Inception, Xception: Esses modelos têm um design mais complexo em comparação com os tradicionais como VGG e ResNet, e podem se situar em diferentes posições nesta lista, dependendo da variante específica.
"""