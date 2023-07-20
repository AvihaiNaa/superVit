from config import CONFIG
from models.train_model import train as train_model
from utils import load_dataset

def train_vit():
    train_loader, validation_loader, test_loader = load_dataset(dataset_name="MNIST", batch_size=CONFIG.VIT.BATCH_SIZE)
    model = train_model(train_loader, validation_loader)
    print("ansqnl")



if __name__ == "__main__":
    train_vit()