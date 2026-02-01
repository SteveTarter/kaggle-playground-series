import os
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

class ExperimentSetup:
    """
    Handles global configuration, seeding, and environment setup.
    """
    def __init__(self, seed=10301, model_name=None):
        self.seed = seed
        self.model_name=model_name

    
    def running_in_kaggle(self) -> bool:
        """
        Heuristics that are true in Kaggle notebooks:
        - Special directories exist (/kaggle/input, /kaggle/working)
        - Env var KAGGLE_KERNEL_RUN_TYPE is set
        - The kaggle_secrets module is available
        """
        try:
            if os.path.isdir('/kaggle/input') and os.path.isdir('/kaggle/working'):
                return True
            if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
                return True
            import kaggle_secrets  # noqa: F401  (only exists in Kaggle)
            return True
        except Exception:
            return False

            
    def set_seeds(self, seed=None) -> int:
        if seed is not None:
            self.seed = seed
            
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        # tf.random.set_seed(seed) # Uncomment before using TensorFlow
        # torch.manual_seed(seed)  # Uncomment before using PyTorch
        print(f'Random seed set to: {self.seed}')
        
        return self.seed

    
    def get_seed(self) -> int:
        return self.seed

        
    def configure_pandas(self, max_cols=None, max_rows=100, float_precision=3):
        pd.set_option('display.max_columns', max_cols)
        pd.set_option('display.max_rows', max_rows)
        pd.options.display.float_format = f'{{:,.{float_precision}f}}'.format

    
    def suppress_warnings(self):
        warnings.filterwarnings('ignore')
        # Specifically suppress the annoying XGBoost warning if needed
        warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
        print('Warnings suppressed.')

    
    def read_training_dataset(self) -> pd.DataFrame:
        data_dir = Path('/kaggle/input/playground-series-s6e1') if self.running_in_kaggle() else Path('data')

        training_df = pd.read_csv(data_dir / 'train.csv')
        print('TRAINING DATASET')
        print('================\n')
        print(training_df.head(5))

        return training_df

    
    def read_test_dataset(self) -> pd.DataFrame:
        data_dir = Path('/kaggle/input/playground-series-s6e1') if self.running_in_kaggle() else Path('data')

        test_df = pd.read_csv(data_dir / 'test.csv')
        print('TEST DATASET')
        print('============\n')
        print(test_df.head(5))

        return test_df

    
    def read_original_dataset(self) -> pd.DataFrame:
        data_dir = Path('/kaggle/input/exam-score-prediction-dataset') if self.running_in_kaggle() else Path('original_data')

        original_df = pd.read_csv(data_dir / 'Exam_Score_Prediction.csv')
        print('ORIGINAL DATASET')
        print('================\n')
        print(original_df.head(5))

        return original_df


    def read_sample_submission_dataset(self) -> pd.DataFrame:
        data_dir = Path('/kaggle/input/playground-series-s6e1') if self.running_in_kaggle() else Path('data')

        submission_df = pd.read_csv(data_dir / 'sample_submission.csv')

        return submission_df
