import os
import torch
import zipfile
import requests
import torchinfo
import torchvision
from torch import nn
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class EngineViT:

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device):
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

    def train_model(self, train_dataloader, test_dataloader):
        EPOCHS=10
        engine = EngineViT(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )
        results = engine.train(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            epochs=EPOCHS
        )
        return results

    @staticmethod
    def transform_pipeline_manually(IMG_SIZE):
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    @staticmethod
    def freeze_base_parameters(pretrained_vit):
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False
        return parameter

    @staticmethod
    def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int= os.cpu_count()
    ):
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
    def set_seeds(seed: int=42):
        """Sets random sets for torch operations.

        Args:
            seed (int, optional): Random seed to set. Defaults to 42.
        """
        # Set the seed for general torch operations
        torch.manual_seed(seed)
        # Set the seed for CUDA torch operations (ones that happen on the GPU)
        torch.cuda.manual_seed(seed)

    @staticmethod
    def prepare_data():
        image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                destination="pizza_steak_sushi")
        TRAIN_DIR = image_path / "train"
        TEST_DIR = image_path / "test"
        IMG_SIZE = 224
        BATCH_SIZE = 32

        manual_transforms = EngineViT.transform_pipeline_manually(IMG_SIZE)
        train_dataloader, test_dataloader, class_names = EngineViT.create_dataloaders(
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            transform=manual_transforms,
            batch_size=BATCH_SIZE
        )
        return train_dataloader, test_dataloader, class_names, TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

    @staticmethod
    def setup_model(class_names, TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, LEARNING_RATE):
            
            COLOR_CHANNELS = 3
            NUMBER_OF_IMAGE_ANALYSED_PER_TIME = 1

            pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

            EngineViT.freeze_base_parameters(pretrained_vit)
            EngineViT.set_seeds()
            pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

            summary(model=pretrained_vit,
                    input_size=(NUMBER_OF_IMAGE_ANALYSED_PER_TIME, COLOR_CHANNELS, IMG_SIZE, IMG_SIZE),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    col_width=20,
                    row_settings=["var_names"]
                )

            vit_transforms = pretrained_vit_weights.transforms()
            train_dataloader_pretrained, test_dataloader_pretrained, _ = EngineViT.create_dataloaders(
                train_dir=TRAIN_DIR,
                test_dir=TEST_DIR,
                transform=vit_transforms,
                batch_size=BATCH_SIZE
            )
            optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=LEARNING_RATE)
            loss_fn = torch.nn.CrossEntropyLoss()

            return pretrained_vit, optimizer, loss_fn, train_dataloader_pretrained, test_dataloader_pretrained
    


    # Plot loss curves of a model
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

    # 1 - take in a trained model
    @staticmethod
    def pred_and_plot_image(model: torch.nn.Module,
                            image_path: str,
                            class_names: List[str],
                            image_size: Tuple[int, int] = (224, 224),
                            transform: torchvision.transforms = None,
                            device: torch.device=device):
        # 2 - open the image with PIL
        img = Image.open(image_path)

        # 3 - create a transform if one desnt exist
        if transform is not None:
            image_transform = transform
        else:
            image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.546, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        ### predict on image ###
        # 4 - make sure the model is on the target device
        model.to(device)

        # 5 - turn the model to eval mode
        model.eval()
        with torch.inference_mode():
            # 6 - transform the iamge and add an extra batch dimension
            transformed_image = image_transform(img).unsqueeze(dim=0) # [batch_size, color_channels, height, width]

            # 7 - make a prediction on the image by passing it to the model
            target_image_pred = model(transformed_image.to(device))

        # 8 - convert the model output logits to prediction probabilities using 'torch.softmax()'
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # 9 - convert the models prediction probabilities to prediction labels using 'torch.argmax()'
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

        # 10 - plot the image with matplotlib and set the title to the prediction label from step 9 and prediction probability from step 8
        plt.figure()
        plt.imshow(img)
        plt.title(f"Prediction: {class_names[target_image_pred_label]} | Probabilite: {target_image_pred_probs.max()}")
        plt.axis(False)
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

    
    train_dataloader, test_dataloader, class_names, train_dir, test_dir,  img_size, batch_size = EngineViT.prepare_data()
    
    model, optimizer, loss_fn, train_dataloader_pretrained, test_dataloader_pretrained = EngineViT.setup_model(class_names, train_dir, test_dir,  img_size, batch_size, LEARNING_RATE=0.01)

    engine = EngineViT(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)

    results = engine.train_model(train_dataloader_pretrained, test_dataloader_pretrained)
    
    EngineViT.plot_loss_curves(results)

    # get the model size in bytes then convert to megabytes
    #pretrained_vit_model_size = Path("models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth").stat().st_size // (1024*1024)
    #print(f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB")

    # making prediction on a custom image
    pred_and_plot_image(model=pretrained_vit, image_path='/content/o5.jpeg', class_names=class_names)