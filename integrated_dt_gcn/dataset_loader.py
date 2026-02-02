"""Dataset loader for Digital Twin datasets."""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys
import os

# Add RP12_paper to path for importing its classes
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RP12_paper'))


class DTDatasetLoader:
    """
    Load Digital Twin datasets from RP12_paper/datasets.
    
    Dataset files:
    - fspl_RMdataset{N}.pkl: 30,000 RadioMap objects (41x41 grid, transmitters, jammers)
    - fspl_PLdataset{N}.pkl: PathLossMapCollection
    - fspl_measurements{N}.pkl: MeasurementCollection (diffs, jammer labels)
    """
    
    def __init__(self, dataset_dir: str):
        """
        Initialize dataset loader.
        
        Args:
            dataset_dir: Path to datasets directory (e.g., '../RP12_paper/datasets')
        """
        self.dataset_dir = Path(dataset_dir)
        
        # Cache loaded data
        self._radio_maps = {}
        self._measurements = {}
        self._path_loss = {}
        
    def load_radio_maps(self, dataset_nr: int = 0) -> List[Any]:
        """
        Load radio map dataset.
        
        Args:
            dataset_nr: Dataset number (0-5, different noise levels)
            
        Returns:
            List of 30,000 RadioMap objects
            Each has: radio_map (41x41), transmitters, jammers
        """
        if dataset_nr not in self._radio_maps:
            path = self.dataset_dir / f"fspl_RMdataset{dataset_nr}.pkl"
            with open(path, 'rb') as f:
                self._radio_maps[dataset_nr] = pickle.load(f)
        return self._radio_maps[dataset_nr]
    
    def load_measurements(self, dataset_nr: int = 0) -> Any:
        """
        Load measurement dataset.
        
        Args:
            dataset_nr: Dataset number
            
        Returns:
            MeasurementCollection with:
            - measurements_diff_list: List of (25,) arrays (DT - actual)
            - jammers_list: List of jammer objects or []
            - meas_x, meas_y: Measurement point coordinates
        """
        if dataset_nr not in self._measurements:
            path = self.dataset_dir / f"fspl_measurements{dataset_nr}.pkl"
            with open(path, 'rb') as f:
                self._measurements[dataset_nr] = pickle.load(f)
        return self._measurements[dataset_nr]
    
    def load_path_loss(self, dataset_nr: int = 0) -> Any:
        """Load path loss map dataset."""
        if dataset_nr not in self._path_loss:
            path = self.dataset_dir / f"fspl_PLdataset{dataset_nr}.pkl"
            with open(path, 'rb') as f:
                self._path_loss[dataset_nr] = pickle.load(f)
        return self._path_loss[dataset_nr]
    
    def get_scenario_count(self, dataset_nr: int = 0) -> int:
        """Get total number of scenarios in dataset."""
        return len(self.load_radio_maps(dataset_nr))
    
    def get_jammed_indices(self, dataset_nr: int = 0) -> List[int]:
        """Get indices of scenarios with jammers."""
        measurements = self.load_measurements(dataset_nr)
        return [i for i, j in enumerate(measurements.jammers_list) if j]
    
    def get_normal_indices(self, dataset_nr: int = 0) -> List[int]:
        """Get indices of scenarios without jammers."""
        measurements = self.load_measurements(dataset_nr)
        return [i for i, j in enumerate(measurements.jammers_list) if not j]
    
    def get_normal_measurements(self, dataset_nr: int = 0) -> List[np.ndarray]:
        """Get measurement diffs from non-jammed scenarios for training."""
        measurements = self.load_measurements(dataset_nr)
        normal_indices = self.get_normal_indices(dataset_nr)
        return [measurements.measurements_diff_list[i] for i in normal_indices]
    
    def get_radio_map_array(self, scenario_idx: int, dataset_nr: int = 0) -> np.ndarray:
        """
        Get radio map as numpy array for a specific scenario.
        
        Args:
            scenario_idx: Index of scenario (0-29999)
            dataset_nr: Dataset number
            
        Returns:
            (41, 41) numpy array of RSS values in dBm
        """
        radio_maps = self.load_radio_maps(dataset_nr)
        return radio_maps[scenario_idx].radio_map
    
    def has_jammer(self, scenario_idx: int, dataset_nr: int = 0) -> bool:
        """Check if scenario has a jammer."""
        measurements = self.load_measurements(dataset_nr)
        return bool(measurements.jammers_list[scenario_idx])
    
    def get_jammer_positions(self, scenario_idx: int, dataset_nr: int = 0) -> List[Tuple[int, int]]:
        """Get jammer positions for a scenario."""
        measurements = self.load_measurements(dataset_nr)
        jammers = measurements.jammers_list[scenario_idx]
        if not jammers:
            return []
        return [tuple(j.tx_pos) for j in jammers]
    
    def get_measurement_diff(self, scenario_idx: int, dataset_nr: int = 0) -> np.ndarray:
        """Get measurement diff array for a scenario."""
        measurements = self.load_measurements(dataset_nr)
        return measurements.measurements_diff_list[scenario_idx]
    
    def get_measurement_points(self, dataset_nr: int = 0) -> Tuple[List[int], List[int]]:
        """Get measurement point coordinates (meas_x, meas_y)."""
        measurements = self.load_measurements(dataset_nr)
        return measurements.meas_x, measurements.meas_y


if __name__ == "__main__":
    # Test loading
    loader = DTDatasetLoader("../RP12_paper/datasets")
    
    print("Loading radio maps...")
    radio_maps = loader.load_radio_maps(0)
    print(f"  Loaded {len(radio_maps)} radio maps")
    print(f"  Sample radio map shape: {radio_maps[0].radio_map.shape}")
    
    print("\nLoading measurements...")
    measurements = loader.load_measurements(0)
    print(f"  Number of scenarios: {len(measurements.measurements_diff_list)}")
    print(f"  Measurement points: {len(measurements.meas_x)}")
    
    print(f"\nJammed scenarios: {len(loader.get_jammed_indices(0))}")
    print(f"Normal scenarios: {len(loader.get_normal_indices(0))}")
    
    # Test specific scenario
    print(f"\nScenario 0 has jammer: {loader.has_jammer(0)}")
    print(f"Scenario 1 has jammer: {loader.has_jammer(1)}")
