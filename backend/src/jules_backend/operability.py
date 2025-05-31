import logging

import numpy as np
import pandas as pd

from functools import lru_cache
from typing import Dict, Tuple, Any

import plotly.graph_objects as go

from jules_backend.model import NautilusModel
from jules_backend.rao import RAOCalculator


OPERABILITY_OUTCOMES = ['mbr_min', 'tension_tdp_min', 'tension_tdp_max', 'normalized_curvature']
THREE_OUTCOMES = ['mbr_min', 'tension_tdp_min', 'tension_tdp_max']
TWO_OUTCOMES = ['tension_tdp_min', 'normalized_curvature']

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def get_model(model_version: str, cable_type: str) -> NautilusModel:
    """Get model for operability table calculations."""
    return NautilusModel(model_version, cable_type)


@lru_cache(maxsize=16)
def get_rao_calculator(vessel: str, vessel_type: str, draught: str) -> RAOCalculator:
    """
    Get RAO calculator for operability table calculations.

    Args:
        vessel: Vessel name
        vessel_type: Vessel type
        draught: Vessel draught

    """
    return RAOCalculator(vessel, vessel_type, draught)


@lru_cache(maxsize=16)
def get_operability_input_grid(vessel: str, vessel_type: str, draught: str,
                               point: Tuple[float, float, float]) -> pd.DataFrame:
    """Get RAO grid for operability table calculations.
    
    Args:
        vessel: Vessel name
        vessel_type: Vessel type
        draught: Vessel draught
        point: tuple of [x, y, z] coordinates relative to COG
        
    Note: The point is relative to COG, not the origin of the vessel. 
    
    Returns:
        pd.DataFrame: DataFrame with RAO values for all wave conditions that will be used in the operability table calculations.
        The dataframe has the following columns:
    """
    rc = get_rao_calculator(vessel, vessel_type, draught)
    rao_grid = rc.rao_table(point)

    # We need to convert the rao_grid to a dataframe
    # The columns are 
    # ['wave_direction', 'wave_period', 'wave_amplitude', 'surge', 'sway', 'heave', 'roll', 'pitch', 'yaw']
    df = rao_grid['amplitudes']
    # We are keeping the translational motions
    columns = rc.wave_params + rc.translational_motions
    df = df[columns]

    # Add a copy of wave_period column as tmax
    tmax_df = pd.DataFrame(df['wave_period'].values, columns=['tmax'])
    df = pd.concat([df, tmax_df], axis=1)

    # Replace translational motions with cable amplitudes  
    column_map = dict(zip(rc.translational_motions, rc.cable_amplitudes))
    df = df.rename(columns=column_map)

    return df


@lru_cache(maxsize=16)
def cache_op_grid_df(model_version: str, cable_type: str, vessel: str, vessel_type: str, draught: str,
                     point: Tuple[float, float, float], features_h: Tuple[Tuple[str, Any], ...]) -> pd.DataFrame:
    _features_d = dict(features_h)
    _model = get_model(model_version, cable_type)
    _op_grid_df = get_operability_input_grid(vessel, vessel_type, draught, point)

    # We already have tmax and cable amplitudes on the ship endA_'xyz'_amplitudes
    _present_features = [v for v in _op_grid_df.columns if v in _model.features]

    # Select features that are passed but are not already available in op_grid_df
    _other_features = [v for v in _features_d.keys() if v not in _present_features]

    _all_features = set(_present_features + _other_features)
    if set(_all_features) != set(_model.features):
        missing = set(_model.features) - set(_all_features)
        if missing:
            msg = f"Missing features: {missing}"
            logger.error(msg)
            raise ValueError(msg)
        else:
            extra = set(_all_features) - set(_model.features)
            if extra:
                msg = f"Unused features: {extra}"
                logger.warning(msg)

    _fill_data = {k: _features_d[k] for k in _other_features}
    _other_features_df = pd.DataFrame([_fill_data for _ in range(_op_grid_df.shape[0])])

    _op_table_input_df = pd.concat([_other_features_df, _op_grid_df], axis=1)

    return _op_table_input_df


def set_operability_table_input_df(model_version: str, cable_type: str,
                                   vessel: str, vessel_type: str, draught: str,
                                   point: Tuple[float, float, float], features: Dict[str, Any]):
    """
    Create operability table input dataframe.
    
    Args:
        model_version: Model version
        cable_type: Cable type
        vessel: Vessel name
        vessel_type: Vessel type
        draught: Vessel draught
        point: tuple of [x, y, z] coordinates for the cable drop point
        features: Dictionary containing features
        
    It is OK if the features dictionary has more features than the model. If the model cannot
    find a feature, it will raise a ValueError.
    
    Note: turning the features dictionary into a tuple of sorted items is used for caching.
    
    Returns:
        pd.DataFrame: DataFrame with operability table input
    
    Raises:
        ValueError: If the features dictionary has missing or unexpected features
    """

    features_h = tuple(sorted(features.items()))

    op_table_input_df = cache_op_grid_df(
        model_version=model_version,
        cable_type=cable_type,
        vessel=vessel,
        vessel_type=vessel_type,
        draught=draught,
        point=point,
        features_h=features_h)

    return op_table_input_df


@lru_cache(maxsize=16)
def cache_predictions(model_version: str, cable_type: str, vessel: str, vessel_type: str, draught: str,
                      point: Tuple[float, float, float], features_h: Tuple[Tuple[str, Any], ...]) -> pd.DataFrame:
    features_d = dict(features_h)

    # Cached
    _op_table_input_df = set_operability_table_input_df(model_version, cable_type,
                                                        vessel, vessel_type, draught, point, features_d)

    # Cached
    _model = get_model(model_version, cable_type)

    _model_input_df = _op_table_input_df[_model.features].to_dict('records')
    _model_outputs = _model.predict(_model_input_df)
    _predictions_df = _model_outputs['predictions'].round(3)
    logger.info(f'Got {len(_predictions_df)} predictions')
    _result_df = pd.concat([_op_table_input_df, _predictions_df], axis=1)
    return _result_df


def predict_outcomes(model_version: str, cable_type: str,
                                   vessel: str, vessel_type: str, draught: str,
                                   point: Tuple[float, float, float], features: Dict[str, Any]):
    """Predict operability parameters from operability table input dataframe.
    1. Create operability table input dataframe
    2. Predict operability parameters
    3. Return dataframe with predictions
    
    Note: turning the features dictionary into a tuple of sorted items is used for caching.
        
    Args:
        model_version: Model version
        cable_type: Cable type
        vessel: Vessel name
        vessel_type: Vessel type
        draught: Vessel draught
        point: tuple of [x, y, z] coordinates for the cable drop point
        features: Dictionary containing features
    
    Returns:
        pd.DataFrame: DataFrame with operability table input
    """
    features_h = tuple(sorted(features.items()))
    point_tuple = tuple(point)

    result_df = cache_predictions(
        model_version=model_version,
        cable_type=cable_type,
        vessel=vessel,
        vessel_type=vessel_type,
        draught=draught,
        point=point_tuple,
        features_h=features_h
    )

    return result_df


def evaluate_operability_all_wave_conditions(predicted_df: pd.DataFrame, mbr_min: float,
                                             tension_tdp_min: float,
                                             tension_tdp_max: float) -> pd.DataFrame:
    """Use predicted outcomes to evaluate if the operability parameters are within the operational range.
    
    Operational range is collected from the frontend_inputs.
    
    Args:
        predicted_df: DataFrame with predicted outcomes
        mbr_min: float, Minimum MBR
        tension_tdp_min: float, Minimum tension at TDP
        tension_tdp_max: float, Maximum tension at TDP
    
    Returns:
        pd.DataFrame: DataFrame with outcomes for all wave conditions
    """
    op_params = {
        'mbr_min': mbr_min,
        'tension_tdp_min': tension_tdp_min,
        'tension_tdp_max': tension_tdp_max,
        'normalized_curvature': 1.0
    }

    # Set outcome boundaries
    operability_range_df = pd.DataFrame([op_params for _ in range(predicted_df.shape[0])])

    delta = predicted_df[THREE_OUTCOMES] - operability_range_df[THREE_OUTCOMES]

    # Predicted mbr_min has to be bigger than op_mbr_min, delta.mbr_min > 0
    # Predicted tension_tdp_min has to be bigger than op_tension_tdp_min, delta.tension_tdp_min > 0
    # Predicted tension_tdp_max has to be smaller than op_tension_tdp_max, delta.tension_tdp_max < 0

    _df = pd.concat(
        [
            delta['mbr_min'] > 0,
            delta['tension_tdp_min'] > 0,
            delta['tension_tdp_max'] < 0,
            predicted_df['normalized_curvature'] < 1.0,
        ],
        axis=1
    )
    three_columns = [f'Operational {c}' for c in THREE_OUTCOMES]
    _df.columns = three_columns + ['Operational normalized_curvature']

    operational = _df[three_columns].sum(axis=1) == 3
    operational.name = 'Operational'

    two_columns = [f'Operational {c}' for c in TWO_OUTCOMES]
    operational_nc = _df[two_columns].sum(axis=1) == 2
    operational_nc.name = 'Operational NC'

    return pd.concat([predicted_df, _df, operational, operational_nc], axis=1)


def get_max_operational_amplitudes(model_version: str, cable_type: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Process operability table to find maximum wave amplitudes where operation is still True
    for each combination of wave_direction and wave_period.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns 'wave_direction', 'wave_period', 
                          'wave_amplitude', and 'Operational'
    
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Matrix where rows are wave_periods, columns are wave_directions,
                          and values are maximum operational wave amplitudes
            - Dict[str, pd.DataFrame]: Dictionary of matrices for each outcome at the maximum operational wave amplitudes
    """
    # Define the wave directions and periods

    model = get_model(model_version, cable_type)
    wave_directions = sorted(df['wave_direction'].unique().tolist())
    wave_periods = sorted(df['wave_period'].unique().tolist())

    # Initialize the result matrices with NaN values
    amplitude_matrix = pd.DataFrame(index=wave_periods, columns=wave_directions)
    outcomes = {}
    for outcome in model.outcomes:
        outcomes[outcome] = pd.DataFrame(index=wave_periods, columns=wave_directions)

    # Group by wave_direction and wave_period
    for direction in wave_directions:
        for period in wave_periods:
            # Filter data for current direction and period
            mask = (df['wave_direction'] == direction) & (df['wave_period'] == period)
            subset = df[mask]

            if subset.empty:
                continue
        
            # Find the maximum amplitude where Operational is True
            operational_subset = subset[subset['Operational']]

            if operational_subset.empty:
                amplitude_matrix.at[period, direction] = 0.
                for outcome in model.outcomes:
                    outcomes[outcome].at[period, direction] = np.nan
            
            else:
                max_amplitude = operational_subset['wave_amplitude'].max()
                # Get the row with maximum amplitude
                max_row = operational_subset[operational_subset['wave_amplitude'] == max_amplitude].iloc[0]

                amplitude_matrix.at[period, direction] = max_amplitude
                for outcome in model.outcomes:
                    outcomes[outcome].at[period, direction] = max_row[outcome]
                
    amplitude_matrix = amplitude_matrix.round(3)
    for outcome in model.outcomes:
        outcomes[outcome] = outcomes[outcome].round(3)

    return amplitude_matrix, outcomes


def plot_operability_heatmap(operability_matrix: pd.DataFrame,
                             title: str = "Maximum Operational Wave Amplitudes") -> go.Figure:
    """
    Create an interactive heatmap visualization of the operability matrix using Plotly.
    
    Args:
        operability_matrix (pd.DataFrame): Matrix of maximum operational wave amplitudes
                                         from get_max_operational_amplitudes()
        title (str): Title for the heatmap plot
    
    Returns:
        plotly.graph_objects.Figure: Interactive heatmap figure
    """
    # Convert values to float and handle NaN values
    z_values = operability_matrix.values.astype(float)

    # Create text array with formatted values
    text_values = np.empty_like(z_values, dtype=object)  # Change to object dtype to handle strings
    for i in range(z_values.shape[0]):
        for j in range(z_values.shape[1]):
            if np.isnan(z_values[i, j]):
                text_values[i, j] = '0.0'
            else:
                # Format with exactly 1 decimal place
                value = float(z_values[i, j])
                text_values[i, j] = f'{value:.1f}'  # Force 1 decimal place

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=operability_matrix.columns,  # wave directions
        y=operability_matrix.index,  # wave periods
        colorscale='RdYlGn',  # 
        colorbar=dict(
            title=dict(text='Wave Amplitude (m)', side='right'),
            tickformat='.1f'  # Format colorbar ticks to 1 decimal place
        ),
        text=text_values,  # Use pre-formatted text values
        texttemplate='%{text}',  # Use exactly what's in our text array
        textfont={"size": 12},  # Increased font size for better readability
        showscale=True,  # Show the color scale
        hoverongaps=False,  # Disable hover on gaps
        hovertemplate='Wave Direction: %{x}<br>Wave Period: %{y}<br>Amplitude: %{text}m<br><extra></extra>'
    ))

    # Update layout with labels and title
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            y=0.93,  # Adjusted title position
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        xaxis_title='Wave Direction (degrees)',
        yaxis_title='Wave Period (s)',
        xaxis=dict(
            tickmode='array',
            ticktext=[f'{int(d)}°' for d in operability_matrix.columns],
            tickvals=operability_matrix.columns
        ),
        yaxis=dict(
            tickmode='array',
            ticktext=[f'{p}s' for p in operability_matrix.index],
            tickvals=operability_matrix.index
        ),
        width=1000,  # Slightly wider
        height=700  # Slightly taller
    )

    return fig


def get_operability_df(model_version: str, cable_type: str, vessel: str, vessel_type: str,
                       draught: str, point: Tuple[float, float, float], features: Dict[str, Any],
                       mbr_min: float, tension_tdp_min: float, tension_tdp_max: float) -> Tuple[
    pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Get the operability dataframe."""

    # Cached
    predictions_df = predict_outcomes(
        model_version=model_version,
        cable_type=cable_type,
        vessel=vessel,
        vessel_type=vessel_type,
        draught=draught,
        point=point,
        features=features)

    # Not cached, fast
    operability_details_df = evaluate_operability_all_wave_conditions(
        predictions_df,
        mbr_min=mbr_min,
        tension_tdp_min=tension_tdp_min,
        tension_tdp_max=tension_tdp_max
    )
    operability_matrix, outcomes = get_max_operational_amplitudes(
        model_version=model_version,
        cable_type=cable_type,
        df=operability_details_df
        )

    return operability_matrix, outcomes


def set_figure_title(vessel: str, vessel_type: str, draught: str, point: Tuple[float, float, float]) -> Tuple[
    str, str]:
    """Set figure title for operability heatmap."""
    vessel_title = f"Vessel: {vessel}, {vessel_type}, {draught}"
    point_title = f"Chute ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})m" if point else ""

    return vessel_title, point_title


def _convert_numpy_types(obj):
    """Convert numpy types to native Python types.
    Used to convert the figure dictionary to a dictionary that can be serialized to JSON.
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    return obj


def get_operability_matrix(model_version: str, cable_type: str,
                           vessel: str, vessel_type: str, draught: str,
                           point: Tuple[float, float, float], features: Dict[str, Any],
                           mbr_min: float, tension_tdp_min: float, tension_tdp_max: float) -> Dict[str, Any]:
    """Get the operability matrix and predicted values.
    
    Args:
        model_version: Model version
        cable_type: Cable type
        vessel: Vessel name
        vessel_type: Vessel type
        draught: Vessel draught
        point: tuple of [x, y, z] coordinates for the cable drop point
        features: Dictionary containing features
        mbr_min: Minimum MBR
        tension_tdp_min: Minimum tension at TDP
        tension_tdp_max: Maximum tension at TDP
    
    Returns:
        dict: Dictionary containing 'data' and 'layout' keys for the Plotly figure
    """

    passed_args = {
        'model_version': model_version,
        'cable_type': cable_type,
        'vessel': vessel,
        'vessel_type': vessel_type,
        'draught': draught,
        'point': point,
        'features': features,
        'mbr_min': mbr_min,
        'tension_tdp_min': tension_tdp_min,
        'tension_tdp_max': tension_tdp_max
    }
    logger.info(f"Passed args to get_operability_matrix: {passed_args}")

    operability_matrix, outcomes = get_operability_df(
        model_version=model_version,
        cable_type=cable_type,
        vessel=vessel,
        vessel_type=vessel_type,
        draught=draught,
        point=point,
        features=features,
        mbr_min=mbr_min,
        tension_tdp_min=tension_tdp_min,
        tension_tdp_max=tension_tdp_max
    )

    params = {
        'columns': operability_matrix.columns.tolist(),
        'index': operability_matrix.index.tolist(),
        'operability': operability_matrix.values.tolist(),
        'outcomes': outcomes
    }

    return _convert_numpy_types(params)


def get_operability_heatmap(model_version: str, cable_type: str,
                            vessel: str, vessel_type: str, draught: str,
                            point: Tuple[float, float, float], features: Dict[str, Any],
                            mbr_min: float, tension_tdp_min: float, tension_tdp_max: float) -> Dict[str, Any]:
    """Create a heatmap visualization of the operability matrix.
    
    Args:
        model_version: Model version
        cable_type: Cable type
        vessel: Vessel name
        vessel_type: Vessel type
        draught: Vessel draught
        point: tuple of [x, y, z] coordinates for the cable drop point
        features: Dictionary containing features
        mbr_min: Minimum MBR
        tension_tdp_min: Minimum tension at TDP
        tension_tdp_max: Maximum tension at TDP
    
    Returns:
        dict: Dictionary containing 'data' and 'layout' keys for the Plotly figure
    """
    input = {
        'model_version': model_version,
        'cable_type': cable_type,
        'vessel': vessel,
        'vessel_type': vessel_type,
        'draught': draught,
        'point': point,
        'features': features,
        'mbr_min': mbr_min,
        'tension_tdp_min': tension_tdp_min,
        'tension_tdp_max': tension_tdp_max
    }
    logger.info(f"Get operability heatmap for {input!r}")

    operability_matrix, outcomes = get_operability_df(**input)

    vessel_title, point_title = set_figure_title(vessel, vessel_type, draught, point)
    title_text = vessel_title
    if point_title:
        title_text += f"<br>{point_title}"

    logger.info(f"Plot operability heatmap for {title_text!r}")
    fig = plot_operability_heatmap(operability_matrix, title_text)

    # Convert the figure to a dictionary and ensure all values are JSON-serializable
    fig_dict = fig.to_dict()

    # Process the figure dictionary to ensure JSON serializability
    processed_fig_dict = _convert_numpy_types(fig_dict)
    logger.info(f"Processed figure dictionary: {processed_fig_dict.keys()!r}")

    return processed_fig_dict
