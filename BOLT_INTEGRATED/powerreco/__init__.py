"""
PowerRECO package — solar and battery sizing with ROI analysis.
"""
from powerreco.solar_sizing   import calculate_solar_sizing, estimate_inverter_cost
from powerreco.battery_sizing  import calculate_battery_sizing
from powerreco.roi_engine      import calculate_roi, assess_feasibility
from powerreco.optimizer       import validate_powerreco_inputs, sweep_sizing_grid, find_optimum_from_existing_run
from powerreco.data_connector  import parse_manager_output, parse_predictor_output, extract_peak_demand_stats

__all__ = [
    "calculate_solar_sizing",
    "estimate_inverter_cost",
    "calculate_battery_sizing",
    "calculate_roi",
    "assess_feasibility",
    "validate_powerreco_inputs",
    "sweep_sizing_grid",
    "find_optimum_from_existing_run",
    "parse_manager_output",
    "parse_predictor_output",
    "extract_peak_demand_stats",
]
