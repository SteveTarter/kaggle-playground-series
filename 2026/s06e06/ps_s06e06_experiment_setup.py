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
    def __init__(
        self, seed=10301,
        model_name=None,
        target='class',
        use_gpu=False,
        perform_rfe=False,
        perform_optuna_tuning=False
    ):
        self._seed = seed
        self._model_name = model_name
        self._target=target
        self._use_gpu = use_gpu
        self._perform_rfe = perform_rfe
        self._perform_optuna_tuning = perform_optuna_tuning

    
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


    def use_gpu(self) -> bool:
        return self._use_gpu

        
    def perform_rfe(self) -> bool:
        return self._perform_rfe

        
    def perform_optuna_tuning(self) -> bool:
        """
        Prevent accidentally running Optuna on Kaggle.
        It burns too much GPU budget.
        """
        return self._perform_optuna_tuning and not self.running_in_kaggle()

        
    def set_seeds(self, seed=None) -> int:
        if seed is not None:
            self._seed = seed
            
        os.environ['PYTHONHASHSEED'] = str(self._seed)
        random.seed(self._seed)
        np.random.seed(self._seed)
        # tf.random.set_seed(_seed) # Uncomment before using TensorFlow
        # torch.manual_seed(_seed)  # Uncomment before using PyTorch
        print(f'Random seed set to: {self._seed}')
        
        return self._seed

    
    def get_seed(self) -> int:
        return self._seed

        
    def configure_pandas(self, max_cols=None, max_rows=100, float_precision=3):
        pd.set_option('display.max_columns', max_cols)
        pd.set_option('display.max_rows', max_rows)
        pd.options.display.float_format = f'{{:,.{float_precision}f}}'.format

    
    def suppress_warnings(self):
        warnings.filterwarnings('ignore')
        # Specifically suppress the annoying XGBoost warning if needed
        warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
        print('Warnings suppressed.')


    def read_dataset(self, dataset_name) -> pd.DataFrame:
        if dataset_name == 'training':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e6') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'train.csv')
        elif dataset_name == 'test':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e6') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'test.csv')
        elif dataset_name == 'original':
            data_dir = Path('/kaggle/input/datasets/fedesoriano/stellar-classification-dataset-sdss17') if self.running_in_kaggle() else Path('original_data')
            df = pd.read_csv(data_dir / 'star_classification.csv')
        elif dataset_name == 'submission':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e6') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'sample_submission.csv')
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # Don't print out the submission dataset
        if dataset_name != 'submission':
            heading = f'{dataset_name.upper()} DATASET'
            print(heading)
            print('=' * len(heading),'\n')
            print(df.head(5))

        return df

    def describe(self):
        print('\nExperiment Setup')
        print('================')
        if self._model_name:
            print(f'Model         : {self._model_name}')

        optuna_overridden = '' if self._perform_optuna_tuning == self.perform_optuna_tuning() else 'Overridden'
        
        print(f'Target        : {self._target}')
        print(f'Use GPU       : {self._use_gpu}')
        print(f'RFE           : {self._perform_rfe}')
        print(f'Optuna Tuning : {self.perform_optuna_tuning()} {optuna_overridden}')
        print(f'On Kaggle     : {self.running_in_kaggle()}')
