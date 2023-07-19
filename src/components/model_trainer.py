# Model training and evaluation

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","bcc_classification.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    