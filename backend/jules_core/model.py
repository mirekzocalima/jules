"""
Model loader that handles loading models from local files or S3.
"""
import boto3
import joblib
import logging
import os

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Dict, Optional

from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger(__name__)

PREDICTOR_ARG_TYPES = Union[Dict[str, float], List[Dict[str, float]]]
S3_BUCKET = 'nautilus-seabed-sg'


def set_model_path(date: str, cable_type: str) -> str:
    return f'models/{date}/model_{cable_type}.joblib'


@lru_cache(maxsize=5)
def list_available_cable_types(version: str = '2025-06-01'):
    """List available cable types for a given version"""
    key = f'models/{version}/'
    s3_path = f's3://{S3_BUCKET}/{key}'
    try:
        logger.info(f"List available cable types for version {version!r} at {s3_path!r}")
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=key)
        if response['KeyCount'] > 0:
            cable_types = [obj['Key'].split('/')[-1].replace('model_', '').replace('.joblib', '') for obj in response['Contents']]
            logger.info(f"Found {len(cable_types)} cable types for version {version!r}: {cable_types}")
            return cable_types
        else:
            logger.info(f"No cable types found for version {version!r}")
            return []
    except ClientError as e:
        logger.error(f"Error downloading from S3: {str(e)}", exc_info=True)
        raise


class NautilusModelLoader:
    """Handles loading models from local files or S3."""

    def __init__(self, version: str, cable_type: str):
        """Initialize the model loader with configuration."""
        self.version = version

        self.available_cable_types = list_available_cable_types(version)

        if cable_type not in self.available_cable_types:
            msg = f"Cable type {cable_type!r} not found in {self.available_cable_types}"
            logger.error(msg)
            raise ValueError(msg)

        self.cable_type = cable_type
        logger.info(f"{self!r}: Loading model")

        self.s3_bucket = S3_BUCKET
        self.s3_model_key = set_model_path(self.version, self.cable_type)

        try:
            base_dir = Path(__file__).parent
        except NameError:
            base_dir = Path(os.getcwd())

        # Add a dot to the model path
        local_model_file =  '.' + self.s3_model_key
        self.model_path = base_dir / local_model_file
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self._details = self._load_model()
        if self.cable_type != '_'.join(self._details['cables']):
            msg = f"Cable mismatch: Passed {self.cable_type}, found {'_'.join(self._details['cables'])}"
            logger.error(msg)
            raise Exception(msg)

        self.model = self._details['model']
        self.scaler = self._details['scaler']
        self.vessel = self._details['vessel']
        self.heave_model = self._details['heave_model']
        self.outcomes = self._details['outcomes']
        self.features = [v for v in self.model.feature_names_in_]

        self._performance_df = None
        self._feature_importance_df = None

        logger.info(f"{self!r}: Model ready")

    def __repr__(self):
        return f"{self.__class__.__name__}(version={self.version!r}, cable_type={self.cable_type!r})"

    def _load_model(self):
        """Load model and scaler from filesystem or S3 if available"""
        if not os.path.exists(self.model_path):
            msg = f"{self!r}: Model file not found at: {self.model_path}. Downloading from S3..."
            logger.info(msg)
            self._download_from_s3()

        logger.info(f"{self!r}: Model file found at {self.model_path}")
        _details = joblib.load(self.model_path)
        logger.info(f"{self!r}: Model details loaded successfully: {_details.keys()}")
        return _details

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _download_from_s3(self):
        """Download model and scaler from S3"""
        try:
            s3_client = boto3.client('s3')

            # Download model
            logger.info(f"{self!r}: Downloading model details from S3: {self.s3_bucket}/{self.s3_model_key}")
            s3_client.download_file(
                self.s3_bucket,
                self.s3_model_key,
                self.model_path
            )
            logger.info(f"{self!r}: Model downloaded successfully")

        except ClientError as e:
            logger.error(f"{self!r}: Error downloading from S3: {str(e)}", exc_info=True)
            raise

    @property
    def performance_df(self):
        if self._performance_df is None:
            self._performance_df = pd.DataFrame(
                self._details['performance']['values'],
                index=self._details['performance']['index'],
                columns=self._details['performance']['columns']
            )
        return self._performance_df

    @property
    def feature_importance_df(self):
        if self._feature_importance_df is None:
            self._feature_importance_df = pd.DataFrame(
                self._details['feature_importance']['values'],
                index=self._details['feature_importance']['index'],
                columns=self._details['feature_importance']['columns']
            )
        return self._feature_importance_df


class NautilusModel(NautilusModelLoader):
    """Handles loading models from local files or S3."""

    def __init__(self, version: str, cable_type: str, ):
        super().__init__(version, cable_type)

    def predict(self, X: PREDICTOR_ARG_TYPES, actual: Optional[PREDICTOR_ARG_TYPES] = None) -> Dict[
        str, Union[pd.DataFrame, None]]:
        """
        Predict outcomes for input samples.
        Args:
            X: dict or list of dicts, each with keys matching self.features
            actual: optional, list or np.ndarray of true values (same shape as predictions)
        Returns:
            np.ndarray of predictions, or dict with errors if actual is provided
        """
        # Accept single dict or list of dicts
        if isinstance(X, dict):
            X = [X]

        X_df = pd.DataFrame(X)

        # Validate input
        X_df = self._validate_and_clean_input(X_df)
        logger.info(f"{self!r}: Input validated and cleaned, shape: {X_df.shape}, columns: {X_df.columns}")

        # Scale
        X_scaled = self.scaler.transform(X_df)
        logger.info(f"{self!r}: Scaled {self.scaler.n_features_in_} features")

        # Predict   
        y_pred = self.model.predict(X_scaled)
        y_pred_df = pd.DataFrame(y_pred, columns=self.outcomes)

        y_true_df = None
        if actual is not None:
            logger.info(f"{self!r}: Outputs provided, type: {type(actual)}, length: {len(actual)}")
            # If actual is a dict, convert to dataframe
            if isinstance(actual, dict):
                actual = [actual]
                logger.info(f"{self!r}: Outputs dict converted to list")

            y_true_df = pd.DataFrame(actual)
            logger.info(f"{self!r}: Truth dataframe created, shape: {y_true_df.shape}")

            if y_pred_df.shape != y_true_df.shape:
                if y_true_df.shape[1] != len(self.outcomes):
                    msg = f"Number of outcomes mismatch: {y_true_df.shape[1]} != {len(self.outcomes)}"
                    logger.error(msg)
                    raise ValueError(msg)
                if y_true_df.shape[0] != X_df.shape[0]:
                    msg = f"Number of samples mismatch: {y_true_df.shape[0]} != {X_df.shape[0]}"
                    logger.error(msg)
                    raise ValueError(msg)

        return {
            'predictions': y_pred_df,
            'truth': y_true_df
        }

    def calculate_error_metrics(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
        residuals = y_true - y_pred
        abs_errors = np.abs(residuals).round(3)
        # Percentage errors only for predictions with abs(y_true) > 1  
        mask = np.abs(y_true) > 1
        percentage_errors = np.zeros_like(residuals, dtype=float)
        percentage_errors = pd.DataFrame(
            np.where(mask, (abs_errors / np.abs(y_true) * 100).round(2), 0.0),
            columns=self.outcomes
        )

        return {
            'residuals': residuals.round(3),
            'percentage_errors': percentage_errors.round(2)
        }

    def _validate_and_clean_input(self, X_df):
        missing = set(self.features) - set(X_df.columns)
        if missing:
            msg = f"{self!r}: Missing features: {missing}"
            logger.error(msg)
            raise ValueError(msg)

        extra = set(X_df.columns) - set(self.features)
        if extra:
            logger.warning(f"{self!r}: Unexpected features: {extra} (will be ignored)")

        X_df = X_df[self.features]  # Only keep expected features

        nan_mask = X_df[self.features].isnull()
        # 1. Systematic missing columns
        all_missing_cols = nan_mask.all(axis=0)
        if all_missing_cols.any():
            cols = list(all_missing_cols[all_missing_cols].index)
            msg = f'{self!r}: All samples are missing values for columns: {cols}'
            logger.error(msg)
            raise ValueError(msg)

        # 2. Drop rows with any NaNs
        if nan_mask.any().any():
            nan_rows = X_df[self.features].index[nan_mask.any(axis=1)].tolist()
            nan_cols = X_df[self.features].columns[nan_mask.any(axis=0)].tolist()
            msg = f"{self!r}: Dropping {len(nan_rows)} samples with missing values in columns: {nan_cols}"
            logger.warning(msg)
            X_df = X_df.dropna(subset=self.features)
            if X_df.empty:
                raise ValueError("All rows have missing values and were dropped.")

        # Optional: check for numeric types
        if not np.issubdtype(X_df[self.features].to_numpy().dtype, np.number):
            raise ValueError("All feature values must be numeric.")

        return X_df
