from config import CONFIG
from models.train_model import train as train_model
from utils import load_dataset
import argparse



parser = argparse.ArgumentParser(description='Process args')
parser.add_argument('--n_epochs', type=int, default=CONFIG.VIT.N_EPOCHS,
                    help="number of epochs")
parser.add_argument('--lr', type=int, default=CONFIG.VIT.LR,
                    help="learning rate")
args = parser.parse_args()


CONFIG.VIT.N_EPOCHS = args.n_epochs

def train_vit():
    train_loader, validation_loader, test_loader = load_dataset(dataset_name="MNIST", batch_size=CONFIG.VIT.BATCH_SIZE)
    model = train_model(train_loader, validation_loader)
    print("ansqnl")



if __name__ == "__main__":
    train_vit()