import os
import torch
import zipfile
import requests
import torchvision
from torch import nn
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

class EngineClass:

    NUM_WORKERS = os.cpu_count()

    def __init__(self, model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 loss_fn: torch.nn.Module, 
                 device: torch.device):
        
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.train()
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc

    def test_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                test_pred_logits = self.model(X)
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    def train(self, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, epochs: int) -> Dict[str, List[float]]:
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(dataloader=train_dataloader)
            test_loss, test_acc = self.test_step(dataloader=test_dataloader)
            print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
        return results
    
    @staticmethod
    def create_dataloaders(train_dir: str, 
                           test_dir: str, 
                           transform: transforms.Compose, 
                           batch_size: int, 
                           num_workers: int=NUM_WORKERS):

        """Creates training and testing DataLoaders.

        Takes in a training directory and testing directory path and turns
        them into PyTorch Datasets and then into PyTorch DataLoaders.

        Args:
            train_dir: Path to training directory.
            test_dir: Path to testing directory.
            transform: torchvision transforms to perform on training and testing data.
            batch_size: Number of samples per batch in each of the DataLoaders.
            num_workers: An integer for number of workers per DataLoader.

        Returns:
            A tuple of (train_dataloader, test_dataloader, class_names).
            Where class_names is a list of the target classes.
            Example usage:
            train_dataloader, test_dataloader, class_names = \
                = create_dataloaders(train_dir=path/to/train_dir,
                                    test_dir=path/to/test_dir,
                                    transform=some_transform,
                                    batch_size=32,
                                    num_workers=4)
        """
        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Get class names
        class_names = train_data.classes

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, class_names

    @staticmethod
    def transform_pipeline_manually(IMG_SIZE):

        return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    @staticmethod   
    def freeze_base_parameters(pretrained_vit):
        # freeze base parameters
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False

        return parameter    
    
    @staticmethod
    def set_seeds(seed: int=42):
        """Sets random sets for torch operations.

        Args:
            seed (int, optional): Random seed to set. Defaults to 42.
        """
        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @staticmethod
    def plot_loss_curves(results):
        """Plots training curves of a results dictionary.

        Args:
            results (dict): dictionary containing list of values, e.g.
                {"train_loss": [...],
                "train_acc": [...],
                "test_loss": [...],
                "test_acc": [...]}
        """
        loss = results["train_loss"]
        test_loss = results["test_loss"]

        accuracy = results["train_acc"]
        test_accuracy = results["test_acc"]

        epochs = range(len(results["train_loss"]))

        plt.figure(figsize=(15, 7))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label="train_loss")
        plt.plot(epochs, test_loss, label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label="train_accuracy")
        plt.plot(epochs, test_accuracy, label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.

    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)

    return image_path


if __name__=="__main__":
        
    image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                            destination="pizza_steak_sushi")

    # Setup directory paths to train and test images
    TRAIN_DIR = image_path / "train"
    TEST_DIR = image_path / "test"
    
    PRE_TRAINED_MODEL_WEITGHTS = torchvision.models.ViT_B_16_Weights.DEFAULT
    PRE_TRAINED_MODEL = torchvision.models.vit_b_16(weights=PRE_TRAINED_MODEL_WEITGHTS).to(device)
    EMBEDDING_SIZE = 768
    LEARNING_RATE = 0.03
    COLOR_CHANNELS = 3
    IMG_SIZE = 224
    BATCH_SIZE = 32 
    EPOCHS = 10
    
    EngineClass.set_seeds()
    
    # Create transform pipeline manually
    manual_transforms = EngineClass.transform_pipeline_manually(IMG_SIZE)

    # Create data loaders
    train_dataloader, test_dataloader, class_names = EngineClass.create_dataloaders(train_dir=TRAIN_DIR, 
                                                                                    test_dir=TEST_DIR, 
                                                                                    transform=manual_transforms, 
                                                                                    batch_size=BATCH_SIZE)
    parameter = EngineClass.freeze_base_parameters(PRE_TRAINED_MODEL)

    # Substituindo a última camada (heads) do modelo
    # A ideia é manter as camadas anteriores (que capturaram recursos genéricos das imagens) 
    # e substituir a última camada para adaptá-la a um novo conjunto de classes.
    PRE_TRAINED_MODEL.heads = nn.Linear(in_features=EMBEDDING_SIZE, out_features=len(class_names)).to(device)

    # get a summary using torchinfo.summary
    summary(model=PRE_TRAINED_MODEL,
            input_size=(BATCH_SIZE, COLOR_CHANNELS, IMG_SIZE, IMG_SIZE),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )

    # get automatic fransforms from pretrained ViT weights
    vit_transforms = PRE_TRAINED_MODEL_WEITGHTS.transforms()

    # setup dataloaders
    train_dataloader_pretrained, test_dataloader_pretrained, class_names = EngineClass.create_dataloaders(train_dir=TRAIN_DIR,
                                                                                                        test_dir=TEST_DIR,
                                                                                                        transform=vit_transforms,
                                                                                                        batch_size=BATCH_SIZE)
    # create optimizer and loss function
    optimizer = torch.optim.Adam(params=PRE_TRAINED_MODEL.parameters(), lr=LEARNING_RATE)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    # Primeiro, você instancia a classe EngineClass
    engine = EngineClass(
        model=PRE_TRAINED_MODEL,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )

    # Então, você chama o método train
    pretrained_vit_results = engine.train(
        train_dataloader=train_dataloader_pretrained,
        test_dataloader=test_dataloader_pretrained,
        epochs=EPOCHS
    )

    print(summary)
    EngineClass.plot_loss_curves(pretrained_vit_results)