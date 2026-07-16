import os
import json
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

class ExperimentSetup:
    """
    Handles global configuration, seeding, and environment setup for PS-S06E07.
    """
    def __init__(
        self,
        seeds=[10301, 42, 2026, 777, 888],
        model_name=None,
        target='health_condition',
        use_gpu=False,
        perform_rfe=False,
        perform_optuna_tuning=False,
        seed=None
    ):
        if seed is not None:
            self._seeds = [seed] if not isinstance(seed, list) else seed
        else:
            self._seeds = [seeds] if not isinstance(seeds, list) else seeds
        self._model_name = model_name
        self._target = target
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
        current_seed = seed if seed is not None else self._seeds[0]
        os.environ['PYTHONHASHSEED'] = str(current_seed)
        random.seed(current_seed)
        np.random.seed(current_seed)
        # tf.random.set_seed(current_seed) # Uncomment before using TensorFlow
        # torch.manual_seed(current_seed)  # Uncomment before using PyTorch
        print(f'Random seed set to: {current_seed}')
        
        return current_seed

    def seed(self) -> list:
        return self._seeds

    def primary_seed(self) -> int:
        return self._seeds[0]

        
    def configure_pandas(self, max_cols=None, max_rows=100, float_precision=3):
        pd.set_option('display.max_columns', max_cols)
        pd.set_option('display.max_rows', max_rows)
        pd.options.display.float_format = f'{{:,.{float_precision}f}}'.format

    
    def suppress_warnings(self):
        warnings.filterwarnings('ignore')
        # Specifically suppress the annoying XGBoost warning if needed
        warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
        print('Warnings suppressed.')


    def read_dataset(self, dataset_name:str) -> pd.DataFrame:
        '''
        Loads datasets from the various sources.
        Adds an 'is_original' flag to help models handle distribution shifts.
        '''
        if dataset_name == 'training':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e7') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'train.csv')
            df['is_original'] = 0
        elif dataset_name == 'test':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e7') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'test.csv')
            df['is_original'] = 0
        elif dataset_name == 'original':
            # Look for the dataset on Kaggle or local directory
            data_dir = Path('/kaggle/input/datasets/ziya07/college-student-health-behavior-dataset') if self.running_in_kaggle() else Path('original_data')
            # If not found at default Kaggle path, try a backup dataset name path
            if self.running_in_kaggle() and not data_dir.exists():
                data_dir = Path('/kaggle/input/enhanced-student-health-behavior-dataset')

            # Use student_health_dataset_50k.csv which aligns well with training data columns
            file_path = data_dir / 'student_health_dataset_50k.csv'
            if not file_path.exists():
                # fallback for other naming conventions or local checks
                file_path = data_dir / 'student_health_dataset_50k.csv'
            
            df = pd.read_csv(file_path)
            df['is_original'] = 1
        elif dataset_name == 'submission':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e7') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'sample_submission.csv')
        elif dataset_name == 'pseudo':
            file_path = Path('/kaggle/input/notebooks/stephentarter/ps-s06e07-pseudo-labeling/data/train_pseudo.csv') if self.running_in_kaggle() else Path('data/train_pseudo.csv')
            df = pd.read_csv(file_path)
            df['is_original'] = 0
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # Cast 'is_original' to bool so it doesn't appear in numeric combos.
        if 'is_original' in df:
            df['is_original'] = df['is_original'].astype('bool')
            
        # Don't print out the submission dataset
        if dataset_name != 'submission':
            heading = f'{dataset_name.upper()} DATASET'
            print(heading)
            print('=' * len(heading),'\n')
            print(df.head(5))

        return df

    
    def get_combined_training_data(self, include_original: bool = True) -> pd.DataFrame:
        """
        Loads the training data and optionally concatenates the original dataset.
        """
        # Load base training data
        train_df = self.read_dataset('training')
        
        if not include_original:
            return train_df

        # Load original data
        orig_df = self.read_dataset('original')
        
        # Safely find overlapping columns to avoid KeyErrors and duplicates
        shared_cols = list(set(train_df.columns).intersection(set(orig_df.columns)))
        
        # Filter both datasets to only include the shared columns
        orig_df = orig_df[shared_cols]
        train_df_filtered = train_df[shared_cols]
        
        # Concatenate
        combined_df = pd.concat([train_df_filtered, orig_df], ignore_index=True)
        
        # Fix the 'id' column for the appended rows
        # This ensures every row has a unique ID for OOF tracking
        combined_df['id'] = combined_df.index
        
        return combined_df

    
    def save_rfe(self, rfe_filename: str, optimal_cols: pd.Index) -> None:
        path = Path(rfe_filename)
        path.write_text(json.dumps(optimal_cols.tolist(), indent=2))
        print(f"Saved {len(optimal_cols)} features to {path}")

    
    def load_rfe(self, rfe_filename: str, X: pd.DataFrame) -> pd.Index:
        path = Path('/kaggle/input/datasets/stephentarter/ps-s06e07-artifacts') / rfe_filename \
               if self.running_in_kaggle() else Path(rfe_filename)
    
        if path.exists():
            optimal_cols = pd.Index(json.loads(path.read_text()))
            print(f"Loaded {len(optimal_cols)} features from {path}")
        else:
            print(f"RFE file '{path}' not found — using all {len(X.columns)} columns.")
            optimal_cols = X.columns
    
        return optimal_cols

    def save_optuna_params(self, optuna_filename: str, params: dict) -> None:
        path = Path(optuna_filename)
        path.write_text(json.dumps(params, indent=2))
        print(f"Saved {len(params)} parameters to {path}")

    
    def load_optuna_params(self, optuna_filename: str) -> dict:
        path = Path('/kaggle/input/datasets/stephentarter/ps-s06e07-artifacts') / optuna_filename \
               if self.running_in_kaggle() else Path(optuna_filename)
    
        if path.exists():
            params = json.loads(path.read_text())
            print(f"Loaded {len(params)} parameters from {path}")
        else:
            print(f"Optuna file '{path}' not found — returning empty dict.")
            params = {}
    
        return params
    
    
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
