from torchdrug import datasets, transforms, data
import matplotlib.pyplot as plt


# A toy protein structure dataset
class EnzymeCommissionToy(datasets.EnzymeCommission):
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/EnzymeCommission.tar.gz"
    md5 = "728e0625d1eb513fa9b7626e4d3bcf4d"
    processed_file = "enzyme_commission_toy.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]



def protein_structure_dataset():

    truncuate_transform = transforms.TruncateProtein(max_length=350, random=False)
    protein_view_transform = transforms.ProteinView(view='residue')
    transform = transforms.Compose([truncuate_transform, protein_view_transform])

    dataset = EnzymeCommissionToy("~/protein-datasets/", transform=transform, atom_feature=None, bond_feature=None)
    train_set, valid_set, test_set = dataset.split()
    print(dataset)
    print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))

    protein = dataset[0]["graph"]
    is_first_two = (protein.residue_number == 1) | (protein.residue_number == 2)
    first_two = protein.residue_mask(is_first_two, compact=True)
    first_two.visualize()
    plt.show()

    return dataset, train_set, valid_set, test_set

dataset, train_set, valid_set, test_set = protein_structure_dataset()

