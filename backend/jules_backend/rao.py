import logging
import numpy as np
import pandas as pd

from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jules_backend.config_loader import load_rao_config


from vessels.loader import NautilusVessels


RAO_PARAMS = load_rao_config()


# Configure logger
logger = logging.getLogger(__name__)


def canonicalize_point(point: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """Canonicalize point coordinates.
    
    Whatever the user passes, return a tuple of 3 floats. If point is None, return (0.0, 0.0, 0.0).
    
    Args:
        point: Point coordinates (x, y, z)
        
    Returns:
        Point coordinates (x, y, z)
    """
    if point is None:
        return (0.0, 0.0, 0.0)
    
    arr = np.asarray(point, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Point must have 3 coordinates")
    return tuple(arr)


def combine_motions(direct_motion: dict, contributions: list) -> dict:
    """Combine direct motion with rotational contributions using complex numbers.
    
    Args:
        direct_motion (dict): Direct motion data with 'amplitude' and 'phase'
        contributions (list): List of complex numbers representing rotational contributions
        
    Returns:
        dict: Combined motion data with 'amplitude' and 'phase'
    """
    # Convert direct motion to complex number
    direct_complex = direct_motion['amplitude'] * np.exp(1j * np.deg2rad(direct_motion['phase']))

    # Combine all contributions
    total_complex = direct_complex + sum(contributions)

    return {
        'amplitude': round(abs(total_complex), 3),
        'phase': round(np.rad2deg(np.angle(total_complex)) % 360, 3)
    }


class RAOBasic:
    """A class to calculates vessel Response Amplitudes and Operator for arbitrary wave parameters.
    
    Description shamelessly stolen from: https://www.calqlata.com/productpages/00081-help.html
    
    Every vessel has six degrees of freedom: surge, sway, heave, roll, pitch, and yaw. The first three are translational 
    motions and the last three are rotational motions.
    
    The three translational motions are surge, sway and heave. Surge is the motion along the vessel's longitudinal 
    axis (forward and backward), sway is the motion perpendicular to the vessel's longitudinal axis (left and right, 
    or port and starboard), and heave is the motion up and down along the vessel's vertical axis. 
    
    The three rotational motions are roll, pitch and yaw. Roll is the rotation of the vessel around its longitudinal 
    axis (bow left and right if standing upright and facing forward), pitch is the rotation of the vessel around its 
    transverse axis (bow up and down if standing upright and facing forward), and yaw is the rotation of the vessel 
    around its vertical axis (bow clockwise and counterclockwise if standing upright and facing forward).
    
    Response Amplitude Operator
    
    The term Response Amplitude Operator comprises two parts: Response Amplitude and Operator, 
    each of which may be described as follows:

    * Response Amplitude: refers to the degree of movement induced in a floating vessel due to a passing 
      hydrodynamic wave and this movement is absolute (or actual). In other words, the response amplitude is the 
      the amount of tranlational and rotational movement in a vessel at sea induced by a passing hydrodynamic wave.
      Translational movement is measured in units of length (m, ft, etc.) and rotational movement is measured in 
      angular units (° or rads).
            
    * Operator: refers to a factor that must be multiplied by a specific value; e.g. wave height (or amplitude) 
      in order to define the absolute (or actual) movement. Here we use the term operator to refer to the 
      amplitude of the wave.

    This class provides functionality to interpolate RA values for any given wave wave_direction
    and wave_period from provided RAO data, using the nearest available data points. The interpolation is 
    performed using bilinear interpolation between the four nearest points in the wave_direction-wave_period space.
    
    The operator is the wave amplitude.
    
    Attributes:
        rao_file: A csv or csv.gz file containing the RAO data with columns for wave_direction,
            wave_period, and various RAO components (surge, sway, heave, roll, pitch, yaw) with
            their respective amplitudes and phases.
            
    Raises:
        ValueError: If required columns are missing from the RAO data or if file format is invalid.
    """
    # List of motions to interpolate, the first three are translational and the last three are rotational
    translational_motions = ['surge', 'sway', 'heave']
    rotational_motions = ['roll', 'pitch', 'yaw']
    motions = translational_motions + rotational_motions
    
    wave_params = ['wave_direction', 'wave_period', 'wave_amplitude']
    cable_amplitudes = ['endA_x_amplitude', 'endA_y_amplitude', 'endA_z_amplitude']
    
    # The RAO dataframe must have the following columns:
    required_columns = [
        'RAODirections', 'RAOPeriodOrFrequency',
        'RAOSurgeAmp', 'RAOSurgePhase',
        'RAOSwayAmp', 'RAOSwayPhase',
        'RAOHeaveAmp', 'RAOHeavePhase',
        'RAORollAmp', 'RAORollPhase',
        'RAOPitchAmp', 'RAOPitchPhase',
        'RAOYawAmp', 'RAOYawPhase'
    ]

    def __init__(self, vessel: str, vessel_type: str, draught: str):
        """Initialize the RAOCalculator with RAO data from a CSV or CSV.GZ file.
        
        Args:
            csv_file (str): Path to the CSV or CSV.GZ file containing RAO data.
                Must have columns:
                - vesselType: Type of vessel
                - draught: Vessel draught description
                - RAODirections: Wave wave_directions in degrees
                - RAOPeriodOrFrequency: Wave wave_periods in seconds
                - RAO{Motion}Amp: Amplitude for each motion (Surge, Sway, etc.)
                - RAO{Motion}Phase: Phase for each motion (Surge, Sway, etc.)
        
        Raises:
            ValueError: If required columns are missing from the RAO data or if file format is invalid.
        """
        self.vessel = vessel
        self.vessel_type = vessel_type
        self.draught = draught
        self.nv = NautilusVessels()
        self.draught_data = self.nv.get_draught_data(vessel, vessel_type, draught)
        self.rao_data = self.draught_data['RAO_df']
        self.rao_directions = self.draught_data['RAODirections']
        self.rao_periods = self.draught_data['RAOPeriods']
        self.rao_origin = np.array(self.draught_data['RAOOrigin'])
        self.results = None

    def _handle_error(self, error: Exception, context: str) -> None:
        """Handle errors consistently across all methods.
        
        Args:
            error: The exception that occurred
            context: Description of the operation that failed
            
        Raises:
            ValueError: With a descriptive error message
        """
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        raise ValueError(f"Failed to {context}: {str(error)}")
    
    def validate_point(self, point: Optional[Tuple[float, float, float]]) -> np.ndarray:
        """Validate point coordinates.
        
        Args:
            point: Optional point coordinates (x, y, z), relative to COG
            
        Raises:
            ValueError: If point coordinates are invalid
        """
        point_tuple = canonicalize_point(point)
        
        return self.cache_validate_point(point_tuple)
    
    @lru_cache(maxsize=128)
    def cache_validate_point(self, point_tuple) -> np.ndarray:
        # point_tuple is always a tuple of 3 floats
        # If the user passes (0, 0, 0), we assume they mean the COG
        if point_tuple == (0.0, 0.0, 0.0):
            point_tuple = self.rao_origin
        point_vessel = np.array(point_tuple) - self.rao_origin
        if not np.allclose(point_vessel, (0, 0, 0)):
            logger.info(f"Passed point: {np.round(point_tuple, 3)}, COG: {np.round(self.rao_origin, 3)}, relative to COG: {np.round(point_vessel, 3)}")
        return point_vessel
    
    def _validate_wave_params(self, wave_direction: float, wave_period: float, wave_amplitude: float) -> None:
        """Validate wave parameters.
        
        Args:
            wave_direction: Wave direction in degrees (0-360)
            wave_period: Wave period in seconds
            wave_amplitude: Wave amplitude in meters
            
        Raises:
            ValueError: If any wave parameter is invalid
        """
        wave_direction_min = RAO_PARAMS['wave_direction']['min']
        wave_direction_max = RAO_PARAMS['wave_direction']['max']
        if not wave_direction_min <= wave_direction <= wave_direction_max:
            raise ValueError(f"Wave direction must be between {wave_direction_min} and {wave_direction_max} degrees")
        
        wave_period_min = RAO_PARAMS['wave_period']['min']
        wave_period_max = RAO_PARAMS['wave_period']['max']
        if not wave_period_min <= wave_period <= wave_period_max:
            raise ValueError(f"Wave period must be between {wave_period_min} and {wave_period_max} seconds")
        
        wave_amplitude_min = RAO_PARAMS['wave_amplitude']['min']
        wave_amplitude_max = RAO_PARAMS['wave_amplitude']['max']
        if not wave_amplitude_min <= wave_amplitude <= wave_amplitude_max:
            raise ValueError(f"Wave amplitude must be between {wave_amplitude_min} and {wave_amplitude_max} meters")

    def _round_value(self, value: Any) -> Any:
        """Helper method to round a value to 3 decimal places if it's a float."""
        if isinstance(value, float):
            return round(value, 3)
        return value

    def _process_dict(self, d: Dict[Any, Any]) -> Dict[Any, Any]:
        """Recursively process a dictionary, rounding float values and keys."""
        result = {}
        for k, v in d.items():
            # Round float keys
            new_key = self._round_value(k) if isinstance(k, float) else k

            # Process values
            if isinstance(v, dict):
                result[new_key] = self._process_dict(v)
            elif isinstance(v, (list, np.ndarray)):
                result[new_key] = self._process_list(v)
            else:
                result[new_key] = self._round_value(v)
        return result

    def _process_list(self, lst: Union[List[Any], np.ndarray]) -> List[Any]:
        """Recursively process a list or numpy array, converting to list and rounding float values."""
        result = []
        for item in lst:
            if isinstance(item, dict):
                result.append(self._process_dict(item))
            elif isinstance(item, (list, np.ndarray)):
                result.append(self._process_list(item))
            else:
                result.append(self._round_value(item))
        return result

    def json(self) -> Dict[str, Any]:
        """Serialize RAO results to JSON-compatible format with 3 decimal places rounding.
        
        Returns:
            Dict[str, Any]: JSON-serializable dictionary with rounded values
            
        Raises:
            ValueError: If no results are available
        """
        if self.results is None:
            raise ValueError("No results available. Call a calculation method first.")
        return self._process_dict(self.results)

    @lru_cache(maxsize=128)
    def _find_nearest_points(self, wave_direction: float, wave_period: float) -> tuple:
        """Find the four nearest points in the wave_direction-wave_period space for interpolation.
        
        Args:
            vessel_type (str): Type of vessel
            draught (str): Vessel draught description
            wave_direction (float): Wave wave_direction in degrees
            wave_period (float): Wave wave_period in seconds
            
        Returns:
            tuple: Contains:
                - List of four nearest points (wave_direction, wave_period) pairs
                - List of corresponding data rows from rao_data
                - Interpolation weights for each point
        """
        # Get the RAO data for this vessel and draught
        data = self.rao_data

        # Get unique sorted wave_directions and wave_periods
        # wave_directions = np.sort(data['RAODirection'].unique())
        # wave_periods = np.sort(data['RAOPeriodOrFrequency'].unique())

        # Find nearest indices
        dir_idx = np.searchsorted(self.rao_directions, wave_direction)
        per_idx = np.searchsorted(self.rao_periods, wave_period)

        # Handle edge cases
        dir_idx = min(max(1, dir_idx), len(self.rao_directions) - 1)
        per_idx = min(max(1, per_idx), len(self.rao_periods) - 1)

        # Get surrounding points
        dir_low, dir_high = self.rao_directions[dir_idx - 1], self.rao_directions[dir_idx]
        per_low, per_high = self.rao_periods[per_idx - 1], self.rao_periods[per_idx]

        # Calculate weights for bilinear interpolation
        dir_weight = (wave_direction - dir_low) / (dir_high - dir_low)
        per_weight = (wave_period - per_low) / (per_high - per_low)

        # Get the four corner points
        points = [
            (dir_low, per_low),
            (dir_low, per_high),
            (dir_high, per_low),
            (dir_high, per_high)
        ]

        # Get corresponding data rows
        rows = []
        for d, p in points:
            mask = np.isclose(data['RAODirection'], d) & np.isclose(data['RAOPeriodOrFrequency'], p, atol=1e-3)
            filtered = data[mask]
            if filtered.empty:
                raise ValueError(f"No RAO data for direction={d}, period={p}")
            rows.append(filtered.iloc[0])

        # Calculate bilinear weights
        weights = [
            (1 - dir_weight) * (1 - per_weight),
            (1 - dir_weight) * per_weight,
            dir_weight * (1 - per_weight),
            dir_weight * per_weight
        ]

        return points, rows, weights

    @lru_cache(maxsize=256)
    def response_amplitudes(self, wave_direction: float, wave_period: float) -> dict:
        """Interpolate RAO values for a given wave_direction and wave_period using bilinear interpolation.
        
        This method performs bilinear interpolation using the four nearest points in the
        wave_direction-wave_period space to estimate RAO values for any arbitrary wave_direction and wave_period.
        The interpolation is performed in the complex plane to properly handle amplitude and phase together.
        
        For wave directions between 180° and 360°, it uses symmetry about the vessel's centerline:
        - For θ > 180°, uses (360° - θ) and:
            - Surge, Heave, Pitch: Same amplitude and phase
            - Sway, Roll, Yaw: Same amplitude but opposite phase (+180°)
        
        Args:
            vessel_type (str): Type of vessel
            draught (str): Vessel draught description
            wave_direction (float): Wave wave_direction in degrees (0-360)
            wave_period (float): Wave wave_period in seconds
            
        Returns:
            dict: Interpolated RAO values with the following structure:
                {
                    'surge': {'amplitude': float, 'phase': float},
                    'sway': {'amplitude': float, 'phase': float},
                    'heave': {'amplitude': float, 'phase': float},
                    'roll': {'amplitude': float, 'phase': float},
                    'pitch': {'amplitude': float, 'phase': float},
                    'yaw': {'amplitude': float, 'phase': float}
                }
                All values are rounded to 3 decimal places.
                
        Note:
            Interpolation is performed in the complex plane to properly handle the
            relationship between amplitude and phase.
        """
        # Normalize wave_direction to 0-360 range
        wave_direction = wave_direction % 360

        # For directions > 180°, use symmetry
        use_symmetry = wave_direction > 180
        if use_symmetry:
            # Map to 0-180° range
            wave_direction = 360 - wave_direction

        # Get interpolation points and weights
        _, rows, weights = self._find_nearest_points(wave_direction, wave_period)

        # Initialize result dictionary
        result = {}

        # Define which motions need phase inversion when using symmetry
        phase_inversion_needed = {
            'surge': False, 'sway': True, 'heave': False,
            'roll': True, 'pitch': False, 'yaw': True
        }

        for motion in self.motions:

            # Get amplitude and phase column names
            amp_col = f'RAO{motion.title()}Amp'
            phase_col = f'RAO{motion.title()}Phase'

            # Convert amplitude and phase to complex numbers for each point
            complex_values = []
            for row in rows:
                amp = row[amp_col]
                phase = np.deg2rad(row[phase_col])  # Convert phase to radians
                complex_values.append(amp * np.exp(1j * phase))

            # Interpolate in complex plane
            complex_result = sum(c * w for c, w in zip(complex_values, weights))

            # If using symmetry and this motion needs phase inversion,
            # multiply by -1 (equivalent to adding 180° to phase)
            if use_symmetry and phase_inversion_needed[motion]:
                complex_result *= -1

            # Convert back to amplitude and phase
            amplitude = abs(complex_result)
            phase = (np.angle(complex_result, deg=True)) % 360

            # Store results with rounded values
            result[motion] = {
                'amplitude': round(amplitude, 3),
                'phase': round(phase, 3)
            }

        return result

    def rao(self, wave_direction: float, wave_period: float, wave_amplitude: float) -> dict:
        """Calculate RAO value at the vessel's center or mass for given wave parameters.
        
        This method multiplies the response amplitudes with the wave amplitude to get
        the actual motion amplitudes. The phase values remain unchanged.
        
        Args:
            wave_direction (float): Wave direction in degrees (0-360)
            wave_period (float): Wave period in seconds
            wave_amplitude (float): Wave amplitude in meters
            
        Returns:
            dict: RAO values with the following structure:
                {
                    'surge': {'amplitude': float, 'phase': float},
                    'sway': {'amplitude': float, 'phase': float},
                    'heave': {'amplitude': float, 'phase': float},
                    'roll': {'amplitude': float, 'phase': float},
                    'pitch': {'amplitude': float, 'phase': float},
                    'yaw': {'amplitude': float, 'phase': float}
                }
                All values are rounded to 3 decimal places.
        """
        # Get response amplitudes
        response = self.response_amplitudes(wave_direction, wave_period)

        # Scale amplitudes by wave amplitude
        result = {}
        for motion, data in response.items():
            result[motion] = {
                'amplitude': round(data['amplitude'] * wave_amplitude, 3),
                'phase': data['phase']
            }

        return result

    def rao_at_point(self, wave_direction: float, wave_period: float, wave_amplitude: float,
                     point: Optional[Tuple[float, float, float]] = None) -> dict:
        """Calculate RAO values at an arbitrary point on the ship.
        
        This method calculates RAO values at a point (x, y, z) measured from the center of mass.
        It takes into account both translational and rotational contributions:
        - Roll contributes to sway and heave
        - Pitch contributes to surge and heave
        - Yaw contributes to surge and sway
        
        We use the following coordinate frame:
            - x: positive forward (bow)
            - y: positive port (left)
            - z: positive downward
        
        And the rotational contributions:
            - Roll: rotation about x (longitudinal axis)
            - Pitch: rotation about y (transverse axis)
            - Yaw: rotation about z (vertical axis)
            
        We use the following sign convention:
            - SurgePositive: 'forward'
            - SwayPositive: 'port'
            - HeavePositive: 'up'
            - RollPositiveStarboard: 'down'
                * If you look forward, a positive roll rotates the vessel so the starboard side goes down (port up).
                * Positive roll (about x) is a rotation from y to z (port to up), which means starboard down.
                * Clockwise when viewed in the direction of the surge
            - PitchPositiveBow: 'down'
                * Positive pitch rotates the bow down (stern up).
                * Positive pitch (about y, port) is a rotation from z to x (up to forward), which means bow down.
                * Clockwise when viewed in the direction of the port side
            - YawPositiveBow: 'port'
                * Positive yaw rotates the bow to port (left), i.e., counterclockwise when viewed from above.
                * Positive yaw (about z, up) is a rotation from x to y (forward to port), which is bow to port.
                * Clockwise when viewed in the direction of the heave, from underneath the vessel
        
        If point is None or rao_origin, returns RAO values at the center of mass.
        
        Note that for the passed point we calcuate the position of the cable drop point
        relative to the vessel's center of gravity.
        
        Args:
            wave_direction (float): Wave direction in degrees (0-360)
            wave_period (float): Wave period in seconds
            wave_amplitude (float): Wave amplitude in meters
            point (tuple, optional): Point coordinates (x, y, z) in meters, where:
                If None or (0, 0, 0), returns RAO values at center of mass.
        
        Returns:
            dict: RAO values at the specified point with the following structure:
                {
                    'surge': {'amplitude': float, 'phase': float},
                    'sway': {'amplitude': float, 'phase': float},
                    'heave': {'amplitude': float, 'phase': float},
                    'roll': {'amplitude': float, 'phase': float},
                    'pitch': {'amplitude': float, 'phase': float},
                    'yaw': {'amplitude': float, 'phase': float}
                }
                All values are rounded to 3 decimal places.
        """

        # Get RAO values at center of mass
        rao_cm = self.rao(wave_direction, wave_period, wave_amplitude)
        # Use point relative to the vessel's center of gravity
        point_vessel = self.validate_point(point)
        
        # Return center of mass RAO values if point is None or (0, 0, 0)
        if np.allclose(point_vessel, (0, 0, 0)):
            return rao_cm

        x, y, z = point_vessel
        
        # Convert rotational phases to radians for calculations
        roll_phase = np.deg2rad(rao_cm['roll']['phase'])
        pitch_phase = np.deg2rad(rao_cm['pitch']['phase'])
        yaw_phase = np.deg2rad(rao_cm['yaw']['phase'])

        # Convert rotational amplitudes to radians for displacement calculations
        roll_amp = np.deg2rad(rao_cm['roll']['amplitude'])
        pitch_amp = np.deg2rad(rao_cm['pitch']['amplitude'])
        yaw_amp = np.deg2rad(rao_cm['yaw']['amplitude'])

        # Calculate rotational contributions to translational motions using complex numbers
        # Roll (φ) contributes to sway and heave
        # Note: y is positive port, z is positive up
        # Rotate about x, so sway gets shorter and heave gets longer; 
        # Take an oar sticking at 45 degrees up relative to the waterline, pointing port.
        # As the boat rolls to the right, the oar will be able to reach higher (increase heave)
        # but will loose horizontal reach (sway)
        sway_from_roll = -roll_amp * z * np.exp(1j * roll_phase)  # Sway gets shorter
        heave_from_roll = roll_amp * y * np.exp(1j * roll_phase)

        # Pitch (θ) contributes to surge and heave
        # Note: x is positive forward, z is positive up
        # Point the oar towards bow at 45 relative to the waterline.
        # As the bow goes down, the oar will be able to reach further forward (surge),
        # but also gets lower (heave)
        surge_from_pitch = pitch_amp * z * np.exp(1j * pitch_phase)  
        heave_from_pitch = -pitch_amp * x * np.exp(1j * pitch_phase)

        # Yaw (ψ) contributes to surge and sway
        # Note: x is positive forward, y is positive port
        # Place the oar horizontally at 45 degrees between the bow and port.
        # As the boat yaws from x to y, the x direction gets shorter (surge), 
        # but the y direction gets longer (sway)
        surge_from_yaw = -yaw_amp * y * np.exp(1j * yaw_phase)  
        sway_from_yaw = yaw_amp * x * np.exp(1j * yaw_phase)  

        # Combine translational and rotational contributions
        result = {}

        # Surge: combine direct surge with contributions from pitch and yaw
        result['surge'] = combine_motions(
            rao_cm['surge'],
            [surge_from_pitch, surge_from_yaw]
        )

        # Sway: combine direct sway with contributions from roll and yaw
        result['sway'] = combine_motions(
            rao_cm['sway'],
            [sway_from_roll, sway_from_yaw]
        )

        # Heave: combine direct heave with contributions from roll and pitch
        result['heave'] = combine_motions(
            rao_cm['heave'],
            [heave_from_roll, heave_from_pitch]
        )

        # Rotational motions remain unchanged
        result['roll'] = rao_cm['roll']
        result['pitch'] = rao_cm['pitch']
        result['yaw'] = rao_cm['yaw']

        return result

    @lru_cache(maxsize=3)
    def _get_wave_parameter_range(self, param_name: str, values: Optional[np.ndarray] = None) -> np.ndarray:
        """Get wave parameter range from config or use provided values."""
        if values is not None:
            return values
        start = RAO_PARAMS[param_name]['min']
        stop = RAO_PARAMS[param_name]['max']
        if param_name != 'wave_direction':
            stop = stop + RAO_PARAMS[param_name]['rao_step']
        step = RAO_PARAMS[param_name]['rao_step']

        return np.arange(start, stop, step)

    @lru_cache(maxsize=16)
    def _cache_rao_table(self, point_tuple):
        wave_directions = self._get_wave_parameter_range('wave_direction')
        wave_periods = self._get_wave_parameter_range('wave_period')
        wave_amplitudes = self._get_wave_parameter_range('wave_amplitude')

        # Create parameter combinations
        directions, periods, amplitudes = np.meshgrid(wave_directions, wave_periods, wave_amplitudes, indexing='ij')
        param_combinations = np.column_stack(
            (directions.ravel(), periods.ravel(), amplitudes.ravel())
        )

        # Initialize arrays for results
        n_combinations = len(param_combinations)
        amplitudes_table = np.zeros((n_combinations, 3 + len(self.motions)))
        phases_table = np.zeros((n_combinations, 3 + len(self.motions)))

        # Copy wave parameters to result tables
        amplitudes_table[:, :3] = param_combinations
        phases_table[:, :3] = param_combinations

        # Process each combination
        for i, (direction, period, amplitude) in enumerate(param_combinations):
            # Get response amplitudes
            response = self.response_amplitudes(direction, period)

            # If point is provided, calculate RAO at that point
            rao_values = self.rao_at_point(direction, period, amplitude, point_tuple)

            # Store results
            for j, motion in enumerate(self.motions):
                amplitudes_table[i, 3 + j] = rao_values[motion]['amplitude']
                phases_table[i, 3 + j] = rao_values[motion]['phase']

        # Create DataFrames
        columns = ['wave_direction', 'wave_period', 'wave_amplitude'] + self.motions
        amplitudes_df = pd.DataFrame(amplitudes_table, columns=columns)
        phases_df = pd.DataFrame(phases_table, columns=columns)

        return {
            'amplitudes': amplitudes_df,
            'phases': phases_df
        }
        
    def rao_table(self, point: Optional[Tuple[float, float, float]] = None) -> Dict[str, pd.DataFrame]:
        """Calculate RAO values for multiple wave parameters using vectorization.
        
        This method creates two tables: one for amplitudes and one for phases,
        with wave parameters as the first three columns.
        
        Args:
            point (tuple, optional): Point coordinates (x, y, z) in meters. Defaults to None.
                
        Returns:
            dict: Dictionary containing two pandas DataFrames:
                - 'amplitudes': DataFrame containing amplitudes for all motions
                - 'phases': DataFrame containing phases for all motions
                Both DataFrames have columns:
                    - wave_direction: Wave direction in degrees
                    - wave_period: Wave period in seconds
                    - wave_amplitude: Wave amplitude in meters
                    - surge: Motion amplitude/phase
                    - sway: Motion amplitude/phase
                    - heave: Motion amplitude/phase
                    - roll: Motion amplitude/phase
                    - pitch: Motion amplitude/phase
                    - yaw: Motion amplitude/phase
        """
        point_tuple = tuple(point)
            
        return self._cache_rao_table(point_tuple)


class RAOCalculator(RAOBasic):
    """A class that applies functionality of RAOBasic methods to FastAPI endpoints.
    
    This class provides a high-level interface for calculating RAO values and generating
    plots, with proper error handling and validation. All methods return the instance
    itself for method chaining, and results can be serialized to JSON using the json()
    method.
    
    Example:
        calculator = RAOCalculator("path/to/rao.csv")
        result = calculator.operators(
            vessel_type="Vessel1",
            draught="Draught1",
            wave_direction=45.0,
            wave_period=8.0,
            wave_amplitude=1.5
        ).json()
    """

    # Plot colors for motions (excluding red which is reserved for wave)
    colors = {
        'surge': '#1f77b4',  # blue
        'sway': '#2ca02c',  # green
        'heave': '#9467bd',  # purple
        'roll': '#ff7f0e',  # orange
        'pitch': '#8c564b',  # brown
        'yaw': '#17becf'  # cyan
    }

    def __init__(self, vessel: str, vessel_type: str, draught: str):
        """Initialize the RAOCalculator with RAO data from a CSV or CSV.GZ file.
        
        Args:
            vessel: Name of vessel
            vessel_type: Type of vessel
            draught: Vessel draught 
        """
        super().__init__(vessel, vessel_type, draught)

    def operators(self, wave_direction: float, wave_period: float, wave_amplitude: float,
                  point: Optional[Tuple[float, float, float]] = None) -> 'RAOCalculator':
        """Calculate RAO values for a single set of parameters.
        
        Args:
            wave_direction: Wave direction in degrees (0-360)
            wave_period: Wave period in seconds
            wave_amplitude: Wave amplitude in meters
            point: Optional point coordinates (x, y, z)
            
        Returns:
            RAOCalculator: Self for method chaining
            
        Raises:
            ValueError: If any parameters are invalid
        """
        try:
            # Validate inputs
            self._validate_wave_params(wave_direction, wave_period, wave_amplitude)

            # Calculate RAO values
            self.results = self.rao_at_point(wave_direction, wave_period, wave_amplitude, point)
            return self

        except Exception as e:
            self._handle_error(e, "calculate RAO values")

    def operators_grid(self, point: Optional[Tuple[float, float, float]] = None,
                       include_phases: bool = False) -> 'RAOCalculator':
        """Calculate RAO values for multiple wave parameters in parallel.
        
        Args:
            point: Optional point coordinates (x, y, z)
            num_processes: Optional number of processes to use for parallelization
            
        Returns:
            RAOCalculator: Self for method chaining
            
        Raises:
            ValueError: If any parameters are invalid
        """
        try:
            
            amplitudes_df, phases_df = self.rao_table(point=point)
            logger.info(f"Amplitudes DataFrame: {amplitudes_df.head()}")
            columns = amplitudes_df.columns.tolist()
            amplitudes = amplitudes_df.values.tolist()

            if include_phases:
                phases = phases_df.values.tolist()
                logger.info(f"Phases DataFrame: {phases_df.head()}")
            else:
                logger.info(f"Phases DataFrame: None")
                phases = None

            self.results = {
                'columns': columns,
                'amplitudes': amplitudes,
                'phases': phases
            }
            return self

        except Exception as e:
            self._handle_error(e, "calculate batch RAO values")

    def plot(self, wave_direction: float, wave_period: float,
             wave_amplitude: float = 1.0,
             time_step: float = 0.1, point: Optional[Tuple[float, float, float]] = None, 
             num_periods: int = 3) -> dict:
        """Generate time series responses for all motion components using Plotly.

        Creates a two-panel plot showing:
        - Upper panel: Wave elevation and translational motions (Surge, Sway, Heave)
        - Lower panel: Wave elevation and rotational motions (Roll, Pitch, Yaw)

        Args:
            wave_direction (float): Wave direction in degrees (0-360)
            wave_period (float): Wave period in seconds
            wave_amplitude (float, optional): Wave amplitude in meters. Defaults to 1.0.
            time_step (float, optional): Time step for the time series in seconds. Defaults to 0.1.
            point (tuple, optional): Point coordinates (x, y, z) in meters, where:
                - x is positive forward
                - y is positive starboard
                - z is positive upward
            num_periods (int, optional): Number of wave periods to plot. Defaults to 3.

        Returns:
            dict: Dictionary containing Plotly figure data and layout
        """
        try:
            # Validate inputs
            self._validate_wave_params(wave_direction, wave_period, wave_amplitude)

            # Calculate duration based on number of periods
            duration = num_periods * wave_period

            # Get time series data using existing method
            ts_data = self.get_time_series(
                wave_direction=wave_direction,
                wave_period=wave_period,
                wave_amplitude=wave_amplitude,
                duration=duration,
                time_step=time_step,
                point=point
            )

            # Add padding to time axis (5% on each side)
            time_padding = duration * 0.05
            x_min = -time_padding
            x_max = duration + time_padding

            # Create figure with two subplots
            fig = make_subplots(
                rows=2, cols=1,
                vertical_spacing=0.12,
                row_heights=[0.5, 0.5]
            )

            # Plot wave in upper panel
            fig.add_trace(
                go.Scatter(
                    x=ts_data['time'], y=ts_data['wave'],
                    name='Wave',
                    line=dict(color='red', dash='dash', width=2),
                    opacity=0.7,
                    legendgroup='trans',
                    showlegend=True
                ),
                row=1, col=1
            )

            # Plot translational motions
            for motion in self.motions[:3]:
                fig.add_trace(
                    go.Scatter(
                        x=ts_data['time'], y=ts_data[motion],
                        name=motion.capitalize(),
                        line=dict(color=RAOCalculator.colors[motion], width=2),
                        legendgroup='trans',
                        showlegend=True
                    ),
                    row=1, col=1
                )

            # Plot wave in lower panel
            fig.add_trace(
                go.Scatter(
                    x=ts_data['time'], y=ts_data['wave'],
                    name='Wave',
                    line=dict(color='red', dash='dash', width=2),
                    opacity=0.7,
                    legendgroup='rot',
                    showlegend=False  # Don't show wave twice in legend
                ),
                row=2, col=1
            )

            # Plot rotational motions
            for motion in self.motions[3:]:
                fig.add_trace(
                    go.Scatter(
                        x=ts_data['time'], y=ts_data[motion],
                        name=motion.capitalize(),
                        line=dict(color=RAOCalculator.colors[motion], width=2),
                        legendgroup='rot',
                        showlegend=True
                    ),
                    row=2, col=1
                )

            # Update layout
            vessel_title, wave_title, point_title = self.set_figure_title(
                wave_amplitude=wave_amplitude,
                wave_period=wave_period,
                wave_direction=wave_direction,
                point=point
            )

            title_text = f"{vessel_title}<br>{wave_title}" + (f"<br>{point_title}" if point_title else "")

            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                title=dict(
                    text=title_text,
                    x=0.5,
                    y=0.93,  # Adjusted title position
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=16)
                ),
                # Make the plot responsive
                height=600,
                width=900,
                autosize=True,  # Allow plot to be responsive
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='LightGray',
                    borderwidth=1,
                    font=dict(size=14),
                    tracegroupgap=5
                ),
                # Add modebar configuration
                modebar=dict(
                    orientation='v',
                    bgcolor='rgba(255, 255, 255, 0.7)',
                    color='rgba(0, 0, 0, 0.7)',
                    activecolor='rgba(0, 0, 0, 1)'
                ),
                # Add hover mode configuration
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=12,
                    font_family='Arial'
                )
            )

            # Update hover templates for better tooltips
            hover_template = (
                    '<b>Time</b>: %{x:.2f} s<br>' +
                    '<b>%{data.name}</b>: %{y:.3f}<br>' +
                    '<extra></extra>'  # Remove secondary box
            )

            for trace in fig.data:
                trace.update(
                    hovertemplate=hover_template,
                    hoverlabel=dict(
                        font_size=12,
                        font_family='Arial'
                    )
                )

            # Update x-axes with bounding box and proper range
            # Top subplot - hide x-axis title
            fig.update_xaxes(
                row=1,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                title_text='',  # Remove x-axis title for top subplot
                tickfont=dict(size=12),
                range=[x_min, x_max],
                tick0=0,
                dtick=wave_period / 2
            )

            # Bottom subplot - show x-axis title
            fig.update_xaxes(
                row=2,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                title_text='Time [s]',
                title_font=dict(size=14),
                tickfont=dict(size=12),
                range=[x_min, x_max],
                tick0=0,
                dtick=wave_period / 2
            )

            # Update y-axes with bounding box
            fig.update_yaxes(
                row=1,
                title_text='Displacement [m]',
                title_font=dict(size=16),  # Increased axis label font size
                tickfont=dict(size=14),  # Increased tick label font size
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                showline=True,  # Add box border
                linewidth=1,  # Border width
                linecolor='black',  # Border color
                mirror=True  # Show border on both sides
            )

            fig.update_yaxes(
                row=2,
                title_text='Rotation [deg]',
                title_font=dict(size=16),  # Increased axis label font size
                tickfont=dict(size=14),  # Increased tick label font size
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(128, 128, 128, 0.2)',
                showline=True,  # Add box border
                linewidth=1,  # Border width
                linecolor='black',  # Border color
                mirror=True  # Show border on both sides
            )

            # Update subplot margins
            fig.update_layout(margin=dict(r=150, t=100, b=60, l=80))  # Slightly reduced top margin

            self.results = fig.to_dict()

            return self

        except Exception as e:
            self._handle_error(e, "generate plot")

    def get_time_series(self, wave_direction: float, wave_period: float,
                         wave_amplitude: float = 1.0,
                         duration: float = 100, time_step: float = 0.1,
                         point: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
        """Generate time series responses for all motion components.

        This method generates time-domain responses for all vessel motions using the
        interpolated RA values. The response for each motion component is calculated
        using the formula: A * cos(ωt + φ), where:
        - A is the motion amplitude (RA amplitude * operator, which is the wave amplitude)
        - ω is the wave frequency (2π/wave_period)
        - φ is the phase angle in radians
        - t is the time vector

        If a point is provided, the translational motions will include
        contributions from rotational motions at that point.

        Args:
            vessel_type (str): Type of vessel
            draught (str): Vessel draught description
            wave_direction (float): Wave direction in degrees (0-360)
            wave_period (float): Wave period in seconds
            wave_amplitude (float, optional): Wave amplitude in meters. Defaults to 1.0.
            duration (float, optional): Duration of the time series in seconds. Defaults to 100.
            time_step (float, optional): Time step for the time series in seconds. Defaults to 0.1.
            point (tuple, optional): Point coordinates (x, y, z) in meters, where:
                - x is positive forward
                - y is positive starboard
                - z is positive upward

        Returns:
            pd.DataFrame: DataFrame containing time series data with columns:
                - time: Time points in seconds
                - wave: Wave elevation in meters
                - surge: Surge motion in meters
                - sway: Sway motion in meters
                - heave: Heave motion in meters
                - roll: Roll motion in degrees
                - pitch: Pitch motion in degrees
                - yaw: Yaw motion in degrees
        """
        # Get RAO values (either at center of mass or at specified point)
        
        rao_values = self.rao_at_point(wave_direction, wave_period, wave_amplitude, point)

        # Create time array
        time = np.arange(0, duration, time_step)

        # Calculate wave frequency
        wave_freq = 2 * np.pi / wave_period

        # Initialize results dictionary with time and wave elevation
        data = {
            'time': time,
            'wave': wave_amplitude * np.cos(wave_freq * time)  # wave amplitude is half the height
        }

        # Calculate time series for each motion component
        for motion_name, motion_data in rao_values.items():
            # Scale amplitude by wave height
            amplitude = motion_data['amplitude']

            # Convert phase to radians
            phase_rad = np.deg2rad(motion_data['phase'])

            # Calculate time series: A * cos(ωt + φ)
            data[motion_name] = amplitude * np.cos(wave_freq * time + phase_rad)

        # Create DataFrame with all time series
        return pd.DataFrame(data)

    def set_figure_title(self, wave_amplitude: float, wave_period: float,
                          wave_direction: float, 
                          point: Optional[Tuple[float, float, float]] = None) -> tuple:
        """Create figure titles for vessel response plots.

        Args:
            vessel_type (str): Type of vessel
            draught (str): Vessel draught description
            wave_amplitude (float): Wave amplitude in meters
            wave_period (float): Wave period in seconds
            wave_direction (float): Wave direction in degrees
            point (tuple, optional): Point coordinates (x, y, z) in meters. Defaults to None.

        Returns:
            tuple: Contains:
                - vessel_title: Title with vessel type and draught
                - wave_title: Title with wave parameters
                - point_title: Title with point coordinates (if point is provided, else empty string)
        """
        vessel_title = f"Vessel: {self.vessel}, {self.vessel_type}, {self.draught}"
        wave_title = f"Wave: A={wave_amplitude:.1f}m, T={wave_period:.2f}s, Dir={wave_direction:.0f}°"
        if point is not None:
            point_title = f"Cable Chute: {tuple(np.round(point, 2))}m from COG"
        else:
            point_title = ""

        return vessel_title, wave_title, point_title
