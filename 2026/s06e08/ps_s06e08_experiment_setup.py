import os
import json
import shutil
import random
import warnings
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

class ExperimentSetup:
    """
    Handles global configuration, seeding, and environment setup for PS-S06E08.
    """
    def __init__(
        self,
        seeds=[10301, 42, 2026, 777, 888],
        model_name=None,
        target='addicted_label',
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


    def use_xkcd_style(self) -> None:
        # First, determine where the font file is located
        dev_font_path = "/home/tarter/repos/kaggle/kaggle-playground-series/resources/xkcd-script.ttf"
        kaggle_font_path = "/kaggle/input/datasets/stephentarter/xkcd-resources/xkcd-script.ttf"
        xkcd_font_path = kaggle_font_path if self.running_in_kaggle() else dev_font_path

        fm.fontManager.addfont(xkcd_font_path)
        font_name = fm.FontProperties(fname=xkcd_font_path).get_name()

        plt.xkcd()
        plt.rcParams['font.family'] = font_name

        print('Using XKCD style')
        
        return


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
        '''
        if dataset_name == 'training':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e8') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'train.csv')
        elif dataset_name == 'test':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e8') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'test.csv')
        elif dataset_name == 'submission':
            data_dir = Path('/kaggle/input/competitions/playground-series-s6e8') if self.running_in_kaggle() else Path('data')
            df = pd.read_csv(data_dir / 'sample_submission.csv')
        elif dataset_name == 'pseudo':
            file_path = Path('/kaggle/input/notebooks/stephentarter/ps-s06e08-pseudo-labeling/data/train_pseudo.csv') if self.running_in_kaggle() else Path('data/train_pseudo.csv')
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        # Don't print out the submission dataset
        if dataset_name != 'submission':
            heading = f'{dataset_name.upper()} DATASET'
            print(heading)
            print('=' * len(heading),'\n')
            print(df.head(5))

        return df

    
    def save_rfe(self, rfe_filename: str, optimal_cols: pd.Index) -> None:
        path = Path(rfe_filename)
        path.write_text(json.dumps(optimal_cols.tolist(), indent=2))
        print(f"Saved {len(optimal_cols)} features to {path}")

    
    def load_rfe(self, rfe_filename: str, X: pd.DataFrame) -> pd.Index:
        path = Path('/kaggle/input/datasets/stephentarter/ps-s06e08-artifacts') / rfe_filename \
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
        path = Path('/kaggle/input/datasets/stephentarter/ps-s06e08-artifacts') / optuna_filename \
               if self.running_in_kaggle() else Path(optuna_filename)
    
        if path.exists():
            params = json.loads(path.read_text())
            print(f"Loaded {len(params)} parameters from {path}")
        else:
            print(f"Optuna file '{path}' not found — returning empty dict.")
            params = {}
    
        return params
    
    
    def upload_artifact(self, file_to_upload:str) -> None:
        
        # Only Upload Artifact if not running on Kaggle
        if not self.running_in_kaggle():
            
            print(f'\nUploading {file_to_upload} to Kaggle dataset...')
            staging_dir = 'kaggle_upload_artifacts'
            existing_dir = 'kaggle_existing_artifacts'
            temp_artifacts = 'kaggle_temp_artifacts'
            
            os.makedirs(staging_dir, exist_ok=True)
            os.makedirs(existing_dir, exist_ok=True)
            os.makedirs(temp_artifacts, exist_ok=True)
            
            try:
                # 1. Download metadata
                metadata_cmd = [
                    "/home/tarter/anaconda3/envs/kaggle_env/bin/kaggle",
                    "datasets", "metadata",
                    "stephentarter/ps-s06e08-artifacts",
                    "-p", staging_dir
                ]
                subprocess.run(metadata_cmd, check=True)
                
                # Convert downloaded metadata to the flat format required for uploading
                metadata_path = f'{staging_dir}/dataset-metadata.json'
                with open(metadata_path, 'r') as f:
                    meta_raw = json.load(f)
                
                info = meta_raw['info']
                flat_meta = {
                    "id": f"{info['ownerUser']}/{info['datasetSlug']}",
                    "title": info['title'],
                    "licenses": [{"name": "CC0-1.0"}]
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(flat_meta, f, indent=2)
                    
                # 2. Download existing files from the dataset to preserve them
                try:
                    download_cmd = [
                        "/home/tarter/anaconda3/envs/kaggle_env/bin/kaggle",
                        "datasets", "download",
                        "stephentarter/ps-s06e08-artifacts",
                        "-p", existing_dir,
                        "--unzip"
                    ]
                    subprocess.run(download_cmd, check=True)
                    
                    # Copy all existing files to the temp folder
                    for file in os.listdir(existing_dir):
                        if file != 'dataset-metadata.json':
                            shutil.copy(f'{existing_dir}/{file}', f'{temp_artifacts}/')
                except Exception as dl_err:
                    print(f"Could not download existing files (normal for first upload): {dl_err}")
                    
                # 3. Copy newly generated local file to the temp folder (overwriting if exist)
                if os.path.exists(file_to_upload):
                    shutil.copy(file_to_upload, f'{temp_artifacts}/')
                    print(f"  Including local file: {file_to_upload}")
                        
                # 4. Copy everything from temp folder to staging directory
                for file in os.listdir(temp_artifacts):
                    shutil.copy(f'{temp_artifacts}/{file}', f'{staging_dir}/')
                    
                # 5. Push version
                upload_cmd = [
                    "/home/tarter/anaconda3/envs/kaggle_env/bin/kaggle",
                    "datasets", "version",
                    "-p", staging_dir,
                    "-m", f'Auto-update {file_to_upload} artifact from local run'
                ]
                subprocess.run(upload_cmd, check=True)
                print("Kaggle artifacts dataset upload completed successfully!")
            except Exception as e:
                print(f"Error uploading to Kaggle: {e}")
            finally:
                for path in (staging_dir, existing_dir, temp_artifacts):
                    if os.path.exists(path):
                        shutil.rmtree(path)


    def save_probabilities(
        self, 
        model_prefix:str, 
        oof_probs: np.ndarray, 
        train_ids: pd.Series, 
        y: pd.Series, 
        filled_mask: np.ndarray, 
        test_probs: np.ndarray, 
        test_ids: pd.Series
    ) -> None:
        
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        
        # Column names: prob_0 (not addicted) and prob_1 (addicted)
        prob_cols = ['prob_0', 'prob_1']
        
        # OOF probabilities
        oof_prob_df = pd.DataFrame(oof_probs, columns=prob_cols)
        oof_df = pd.concat([
            pd.DataFrame({'id': train_ids.values, 'target': y.values}),
            oof_prob_df
        ], axis=1)
        oof_df = oof_df[filled_mask]
        oof_df.to_csv(f'{output_dir}/{model_prefix}_oof_probs.csv', index=False)
        
        # Test probabilities
        test_prob_df = pd.concat([
            pd.DataFrame({'id': test_ids.values}),
            pd.DataFrame(test_probs, columns=prob_cols)
        ], axis=1)
        test_prob_df.to_csv(f'{output_dir}/{model_prefix}_test_probs.csv', index=False)
        
        print(f'Saved for ensembling:\n - {output_dir}/{model_prefix}_oof_probs.csv\n - {output_dir}/{model_prefix}_test_probs.csv')

        
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
