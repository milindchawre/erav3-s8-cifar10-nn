import torch

class Config:
    # Dataset
    DATA_ROOT = "./data"
    NUM_CLASSES = 10
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 75
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 1e-5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model
    MODEL_SAVE_PATH = "model.pth" 
    
    # OneCycleLR parameters
    ONE_CYCLE_LR = True
    MAX_LR = 0.02
    DIV_FACTOR = 5
    PCT_START = 0.1