import os
import logging
import time
from typing import Dict, List, Tuple, Optional

import yaml
import pandas as pd

# Local hardware modules
from src.pump_controller_ble import PumpControllerBLE
from src.pump_controller import PumpController
from src.temperature_controller import PeltierModule
from src.electrochem_system import ECMeasurements

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename="optimiser.log",
    filemode="a",
)

class scheduler:
    def __init__(self, config_path: str, recipe_path: Optional[str] = None) -> None:
        """
        Initialize the scheduler for hardware control.
        Loads configuration from YAML, sets up pump controllers (A and B), temperature controller,
        potentiostat, and measurement system. Maps chemicals to pumps and sets priming state.
        
        Args:
            config_path (str): Path to YAML configuration file.
            recipe_path (str): Path to YAML recipe file. Optional.
        """

        self.cfg = self._load_config(config_path)

        self.recipe_path = recipe_path
        if self.recipe_path:
            # Load most recent from yaml
            self.recipe = self._load_config(self.recipe_path)

        # Temperature controller
        pel_cfg = self.cfg["communication"]["temperature_controller"]
        self.tec = PeltierModule(COM=pel_cfg["port"], baud=pel_cfg["baud"], sim=pel_cfg["mock"])

        # Potentiostat
        squid_cfg = self.cfg["communication"]["squidstat"]

        # Measurements helper (for data/paths, post-processing, etc.)
        meas_cfg = self.cfg.get("measurements", {})
        self.cell = ECMeasurements(
            squid_port=squid_cfg["port"], 
            instrument=squid_cfg["instrument"],
            results_path=meas_cfg.get("results_root", "./results") ,
            channel=squid_cfg["channel"],
            squid_sim=squid_cfg["mock"]
            )
        
        # Pumps (two 4‑channel controllers → 8 chemicals total)
        pumpA_cfg = self.cfg["communication"]["pump_controller_A"]
        pumpB_cfg = self.cfg["communication"]["pump_controller_B"]

        # BLE or Serial?
        if pumpA_cfg["ble"]:
            pumpA = PumpControllerBLE(device_name=pumpA_cfg["ble_name"], sim=pumpA_cfg["mock"], timeout=pumpA_cfg["timeout"])
        else:
            pumpA = PumpController(COM=pumpA_cfg["port"], baud=pumpA_cfg["baud"], sim=pumpA_cfg["mock"])

        if pumpB_cfg["ble"]:
            pumpB = PumpControllerBLE(device_name=pumpB_cfg["ble_name"], sim=pumpB_cfg["mock"], timeout=pumpB_cfg["timeout"])
        else:
            pumpB = PumpController(COM=pumpB_cfg["port"], baud=pumpB_cfg["baud"], sim=pumpB_cfg["mock"])

        self.pumpA = pumpA
        self.pumpB = pumpB

        # To be populated via protocol
        self.test_cell_volume = 2.5
        self.electrolyte_volume = 1.5
        self.mixing_time = 0
        self.mixing_cycles = 0

        # Populate temperature related constants
        temp_cfg = self.cfg.get("temperature", {})
        self.tec.allowable_error = temp_cfg["tolerance_C"]
        self.tec.steady_state = temp_cfg["steady_s"]
        self.tec.timeout = temp_cfg["timeout_s"]
        self.cell.settle_time = temp_cfg["settle_s"]

        # Map chemicals to (controller, pump_index)
        # Example entry in YAML:
        # chemicals:
        #   Na2SO4: { controller: A, pump_index: 0, prime_ml: 6.0 }
        self.chem_map = self.cfg["chemicals"]

        # To be populated after experiments
        self.latest_eis_ids = None
        self.latest_cv_ids = None

        # Primed state
        self._primed = False

        # Action map for protocol yaml
        self.ACTIONS = {
            "Prime": lambda p: self.smart_priming(),
            "Deprime": lambda p: self.smart_priming(just_deprime=True),
            "CleanSystem": lambda p: self.system_flush(
                cleaning_agent=p.get("cleaning_agent", "Ethanol"),
                flushing_agent=p.get("flushing_agent", "Milli-Q"),
                cleaning_temperature=p.get("temperature_C"),
                cleaning_time=p.get("cleaning_time_s", 60),
            ),
            "MakeMixture": lambda p: self.make_mixture(),
            "Mix": lambda p: self.mix_electrolyte(),
            "TransferToCell": lambda p: self.transfer_to_cell(
                mixing_time = p.get("mixing_time", 10),
                mixing_cycles = p.get("mixing_cycles", 10),
            ),
            "EIS": lambda p: self.run_temperature_sweep_with_eis(
                setpoints_C = p.get("temperatures_C", [25]),
                freq_start_Hz = p.get("freq_start_Hz", 85000),
                freq_stop_Hz = p.get("freq_end_Hz", 1),
                voltage_amplitude = p.get("amplitude_v", 0.01),
                voltage_bias = p.get("bias_v", 0),
                points_per_decade = p.get("ppd", 20),
                measurements = p.get("measurements_per_temp", 1),
                keep_on = p.get("keep_on", False),
            ),
            "CV": lambda p: self.run_temperature_sweep_with_cv(
                setpoints_C = p.get("temperatures_C", [25]),
                first_limit_V = p.get("first_limit_V", 1.5),
                second_limit_V = p.get("second_limit_V", -1.5),
                scan_rate = p.get("scan_rate", 0.1),
                sampling_interval_s = p.get("sampling_interval_s", 0.01),
                cycles = p.get("cycles", 1),
                measurements = p.get("measurements_per_temp", 1),
                keep_on = p.get("keep_on", False),
            ),
            "TransferToWaste": lambda p: self.transfer_to_waste(),
            "ClearSystem": lambda p: self.clear_system(),
        }
    

    # -------------------- Public API --------------------
    def show_message(self, msg: str) -> None:
        self.pumpB.display_oled_message(msg)

    def smart_priming(self, just_deprime: bool = False) -> None:
        """
        Smart priming solution for all chemical lines by transferring to waste. 
        Depriming always occurs first to prevent excess liquid in mixing chamber.
        
        Args:
            just_deprime (bool): Set to true to block repriming. Defaults to False.
        """

        self.show_message("<-- Depriming")
        flow_rate = self.cfg["pumps"].get("priming_flowrate", 0.05)

        # Build priming arrays
        ml_A = [0.0, 0.0, 0.0, 0.0]
        ml_B = [0.0, 0.0, 0.0, 0.0]

        for chem, meta in self.chem_map.items():
            ctl, idx = self._where(chem)
            prime_ml = meta.get("prime_ml", 0.0)

            if prime_ml <= 0 or prime_ml > 10:
                log.warning(f"Check prime volume for {chem}, skipping for now..")
                continue

            if ctl == "A":
                ml_A[idx-1] = prime_ml
            elif ctl == "B":
                ml_B[idx-1] = prime_ml
            else:
                raise ValueError(f"Unknown controller: {ctl}")

        log.info("Depriming all chemicals simulateously..")
        log.info(f"Controller A: {ml_A} ml")
        log.info(f"Controller B: {ml_B} ml")

        # Deprime
        self.pumpA.multi_pump([-x for x in ml_A], flow_rate=flow_rate, check=False)
        self.pumpB.multi_pump([-x for x in ml_B], flow_rate=flow_rate, check=False)
        self._wait_for_responses()

        self._primed = False

        if not just_deprime:
            log.info("Repriming all chemicals simulateously..")
            self.show_message("--> Repriming")

            # Reprime
            self.pumpA.multi_pump(ml_A, flow_rate=flow_rate, check=False)
            self.pumpB.multi_pump(ml_B, flow_rate=flow_rate, check=False)
            self._wait_for_responses()

            self._primed = True
            self.clear_system()

            self.show_message("System Reprimed!")

        else:
            self.show_message("System Deprimed!")
    
    def make_mixture(self) -> None:
        """
        Dose a mixture of chemicals according to the provided recipe.
        Validates chemicals, updates electrolyte name, primes lines if needed, and doses each chemical
        using the correct pump controller and channel. Waits for hardware responses after dosing.
        
        Args:
            recipe_ml (Dict[str, float]): Mapping of chemical names to volumes (ml) to dose.
        """
        if self.recipe_path:
            # Load most recent from yaml
            self.recipe = self._load_config(self.recipe_path)
        else:
            raise RuntimeError("Can't make mixutre: no recipe_path given!")

        # Validate chemicals exist
        unknown = [k for k in self.recipe.keys() if k not in self.chem_map]
        if unknown:
            raise ValueError(f"Unknown chemical(s): {unknown}")
        
        # Update electrolyte name using mixture
        self.cell.electrolyte = self._construct_mixture_title(self.recipe)

        # Ensure lines are primed before first actual dosing
        if not self._primed:
            log.info("System not primed. Priming and cleaning lines first..")

            self.smart_priming()
            self.system_flush()
        else:
            log.info("System already primed. Continuing..")

        # Build per‑controller pump seconds arrays (len=4 each)
        ml_A = [0.0, 0.0, 0.0, 0.0]
        ml_B = [0.0, 0.0, 0.0, 0.0]

        for chem, vol_ml in self.recipe.items():
            ctl, idx = self._where(chem)

            if vol_ml < 0:
                vol_ml = 0

            if ctl == "A":
                ml_A[idx-1] = vol_ml
            else:
                ml_B[idx-1] = vol_ml

        total_ml = sum(x for x in ml_A) + sum(x for x in ml_B)
        if total_ml > self.electrolyte_volume:
            log.error(f"Total mixture volume is greater than expected electrolyte volume: {total_ml} > {self.electrolyte_volume}")

        count = sum(1 for x in ml_A if x != 0) + sum(1 for x in ml_B if x != 0)
        self.show_message(f"--> Mixing {count} Electrolytes")

        flow_rate = self.cfg["pumps"].get("mixing_flowrate", 0.05)

        # Fire both controllers (start with controller A, then B)
        # First with check=False to fire both simulatenously, then check afterwards
        log.info("Mixing all chemicals simulateously..")
        log.info(f"Controller A: {ml_A}")
        log.info(f"Controller B: {ml_B}")

        self.pumpA.multi_pump(ml_A, flow_rate=flow_rate, check=False)
        self.pumpB.multi_pump(ml_B, flow_rate=flow_rate, check=False)
        self._wait_for_responses()

    def mix_electrolyte(self, mixing_time: int = 10, mixing_cycles: int = 10):
        """
        Perform a number of mixing cycles by pumping forwards and back through tubing. Wait for period of time for mixture to settle.
        """

        log.info(f"Mixing content of chamber {mixing_cycles} times..")
        vol = self.cfg["system"].get("mix_to_cell_ml", 0)
        pwm = self.cfg["pumps"].get("mixing_pwm", 100)

        self.show_message(f"--> Mixing Electroyte X{mixing_cycles}")

        for _ in range(mixing_cycles):
            # Forward
            self._transfer_pump(ctl="A", pump_index=1, volume_ml=vol * 0.7, pwm=pwm, check=True)
            # Back
            self._transfer_pump(ctl="A", pump_index=1, volume_ml=vol, pwm=-pwm, check=True)

        # Final back flow to be sure
        self._transfer_pump(ctl="A", pump_index=1, volume_ml=vol, pwm=-pwm, check=True)

        log.info(f"Waiting for {mixing_time}s for mixture to settle..")
        self.show_message(f"--> Waiting for {mixing_time}s")
        time.sleep(mixing_time)

    def transfer_to_cell(self, check: bool = True, cell_no: int = 1):
        """
        Transfer the mixed solution from the mixing chamber to the test cell.
        Uses pump controller A, channel 1, and adds any extra volume specified in config.
        
        Args:
            check (bool): Whether to check hardware response after transfer.
            cell_no (int): Transfer to cell 1 or cell 2. Defaults to 1.
        """
        if cell_no < 1 or cell_no > 2:
            raise ValueError(f"Cell number provided is incompatible: {cell_no}")

        log.info(f"Transferring {self.test_cell_volume}ml to cell #{cell_no}..")
        extra_vol = self.cfg["system"].get("mix_to_cell_ml", 0)
        pwm = self.cfg["pumps"].get("transfer_pwm", 100)

        if check is True:
            self.show_message(f"--> Transferring to Cell #{cell_no}")

        self._transfer_pump("A", cell_no, self.test_cell_volume + extra_vol, pwm, check)
        
    def transfer_to_waste(self, check: bool = True):
        """
        Transfer the solution from the test cell to waste.
        Uses pump controller B, configurable waste channel, and adds any extra volume specified in config.
        
        Args:
            check (bool): Whether to check hardware response after transfer.
        """
        extra_vol = self.cfg["system"].get("cell_to_waste_ml", 0)
        waste_no = self.cfg["system"].get("waste_no", 1)
        pwm = self.cfg["pumps"].get("waste_pwm", 100)

        if check is True:
            self.show_message(f"--> Transferring to Waste #{waste_no}")

        log.info(f"Transferring {self.test_cell_volume}ml to waste #{waste_no}..")

        self._transfer_pump("B", waste_no, self.test_cell_volume + extra_vol, pwm, check)

    def clear_system(self):
        """
        Transfer contents of mixing chamber straight to waste.
        """
        self.show_message("--> Clearing System")

        self.transfer_to_waste()
        self.transfer_to_cell()
        self.transfer_to_waste()

    def system_flush(self, cleaning_agent: str = "Ethanol", flushing_agent: str = "Milli-Q", cleaning_time: float = 30, cleaning_temperature: float = None):
        """
        Heated cleaning of test cell, with rinses and prolonged cleaning with cleaning agent. 
        
        Args:
            cleaning_agent (str): Name of cleaning agent (e.g. Ethanol), to match chemical name in config file.
            flushing_agent (str): Name of flushing agent (E.g. Milli-Q Water), to match chemical name in config file.
        """
        flush_volume = self.test_cell_volume
        flow_rate = self.cfg["pumps"].get("priming_flowrate", 0.05)

        log.info("Beginning heated cleaning procedure..")

        # Heated cleaning with agent
        if cleaning_temperature:
            self.tec.set_temperature(cleaning_temperature)

        self.clear_system()

        self.show_message("--> Rinsing Cell")

        # Quick flush to clear any salt from lines
        log.info(f"Rinsing with {flushing_agent}.")
        self._single_dose(flushing_agent, flush_volume, flow_rate)
        
        self.clear_system()

        self.show_message("--> Cleaning Cell")

        log.info(f"Cleaning with {cleaning_agent}.")
        self._single_dose(cleaning_agent, flush_volume, flow_rate)
        self.transfer_to_cell()

        self.show_message(f"Waiting for {cleaning_time}s")
        log.info(f"Cleaning for {cleaning_time}s.")

        if not self.cell.sim:
            time.sleep(cleaning_time)

        self.transfer_to_waste()

        self.show_message("--> Rinsing Cell")

        # Final flush
        log.info(f"Final rinsing with {flushing_agent}.")
        self._single_dose(flushing_agent, flush_volume, flow_rate)
        self.clear_system()

        log.info(f"Waiting for another {cleaning_time}s for residue to evaporate.")
        self.show_message(f"Waiting for {cleaning_time}s @ {self.tec.get_t1_value():.1f}C")
        
        if not self.cell.sim:
            time.sleep(cleaning_time)

        self.tec.clear_run_flag()

    def run_temperature_sweep_with_eis(
        self,
        setpoints_C: List[float],
        freq_start_Hz: float,
        freq_stop_Hz: float,
        voltage_amplitude: float,
        voltage_bias: float,
        points_per_decade: int,
        measurements: int,
        keep_on: bool = False,
    ) -> None:
        """
        Run a temperature sweep with EIS (Electrochemical Impedance Spectroscopy) measurements.
        For each temperature setpoint, waits for temperature controller to reach setpoint, then runs EIS
        experiment using the potentiostat and measurement system. Handles hardware errors and clears run flag.
        
        Args:
            setpoints_C (List[float]): List of temperature setpoints in Celsius.
            freq_start_Hz (float): Starting frequency for EIS.
            freq_stop_Hz (float): Ending frequency for EIS.
            voltage_amplitude (float): Amplitude of voltage for EIS.
            voltage_bias (float): Bias voltage for EIS.
            points_per_decade (int): Number of points per frequency decade.
            measurements (int): Number of measurements per temperature.
        """

        self.latest_eis_ids = []
        
        for T in setpoints_C:
            log.info(f"Waiting until temperature = {T:.1f} C")
            self.show_message(f"Target: {self.tec.get_t1_value():.1f} -> {T:.1f}C")

            if not self.tec.wait_until_temperature(T, show_temperature_fn=self.show_message):
                raise RuntimeError("Temperature regulation failed!")
            
            self.show_message(f"Collecting EIS Data @ {self.tec.get_t1_value():.1f}C")
        
            # Build and run the electrochemical experiment
            try:
                id = self.cell.perform_EIS_experiment(
                    start_frequency = freq_start_Hz,
                    end_frequency = freq_stop_Hz,
                    points_per_decade = points_per_decade,
                    voltage_amplitude = voltage_amplitude,
                    voltage_bias = voltage_bias,
                    target_temperature = T,
                    get_temperature_fn = self.tec.get_t1_value,
                    measurements = measurements
                )
                
                self.latest_eis_ids.append(id)

            except Exception as e:
                log.error(f"EIS measurement failed: {e}")

        # Clear run flag on temperature controller
        if not keep_on:
            self.tec.clear_run_flag()

    def run_temperature_sweep_with_cv(
        self,
        setpoints_C: List[float],
        first_limit_V: float,
        second_limit_V: float,
        scan_rate: float,
        sampling_interval_s: float,
        cycles: int,
        measurements: int,
        keep_on: bool = False,
    ) -> None:
        """
        Run a temperature sweep with CV (Cyclic voltammetry) measurements.
        For each temperature setpoint, waits for temperature controller to reach setpoint, then runs CV
        experiment. OCP defines start and end voltages. Handles hardware errors and clears run flag.
        
        Args:
            setpoints_C (List[float]): List of temperature setpoints in Celsius.
            first_limit_V (float): First voltage limit for CV.
            second_limit_V (float): Second voltage limit for CV.
            scan_rate (float): Speed of voltage change (V/s).
            sampling_interval_s (float): Interval between voltage measurements (s)
            cycles (int): Number of cycles.
            measurements (int): Number of measurements per temperature.
        """

        self.latest_cv_ids = []
        
        for T in setpoints_C:
            log.info(f"Waiting until temperature = {T:.1f} C")
            self.show_message(f"Target: {self.tec.get_t1_value():.1f} -> {T:.1f}C")

            if not self.tec.wait_until_temperature(T, show_temperature_fn=self.show_message):
                raise RuntimeError("Temperature regulation failed!")
            
            self.show_message(f"Collecting CV Data @ {self.tec.get_t1_value():.1f}C")
        
            # Build and run the electrochemical experiment
            try:
                id = self.cell.perform_CV_experiment(
                    first_voltage_limit = first_limit_V,
                    second_voltage_limit = second_limit_V,
                    scan_rate = scan_rate,
                    sampling_interval = sampling_interval_s,
                    cycles = cycles,
                    target_temperature = T,
                    get_temperature_fn = self.tec.get_t1_value,
                    measurements = measurements
                )
                
                self.latest_cv_ids.append(id)

            except Exception as e:
                log.error(f"CV measurement failed: {e}")

        # Clear run flag on temperature controller
        if not keep_on:
            self.tec.clear_run_flag()
    
    def run_protocol(self, filename: str) -> None:
        """
        Execute a user-defined protocol from a YAML file (v1 schema).

        Args:
                filename (str): Relative path to protocol yaml.
        """

        start = time.time()

        with open(filename, "r") as f:
            proto = yaml.safe_load(f) or {}

        # --- Metadata overrides (optional) ---
        meta = proto.get("metadata", {}) or {}
        if hasattr(self, "cell"):
            self.cell.user = meta.get("user", getattr(self.cell, "user", "Unknown"))
            self.cell.project = meta.get("project", getattr(self.cell, "project", "Unknown"))
            self.cell.electrolyte = meta.get("electrolyte", getattr(self, "electrolyte", "Unknown"))
            if "cell_constant" in meta:
                self.cell.cell_constant = meta["cell_constant"]
        if "cell_volume" in meta:
            self.test_cell_volume = float(meta["cell_volume"])
        if "electrolyte_volume" in meta:
            self.electrolyte_volume = float(meta["electrolyte_volume"])

        repeats = int(proto.get("repeats", 1) or 1)
        steps = proto.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("protocol.steps must be a LIST of step objects. Example: steps: [ {action: Prime}, ... ]")

        # --- Execute ---
        for rep in range(repeats):
            self.show_message(f"--> Protocol pass {rep+1}/{repeats}")
            for i, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    raise ValueError(f"Step #{i} must be a mapping/object with at least an 'action'. Got: {step!r}")
                action = step.get("action")
                if action not in self.ACTIONS:
                    raise ValueError(f"Unknown action '{action}'. Valid: {list(self.ACTIONS.keys())}")
                params = {k: v for k, v in step.items() if k != "action"}
                self.ACTIONS[action](params)

        end = time.time() - start
        minutes = round(end / 60, 2)

        self.show_message(f"Experiment Time = {minutes}m")
        log.info(f"[TIMER] Experiment completed in {minutes}mins.")         

    def update_yaml_volumes(self, values: dict) -> None:
        """
        Update 'recipe_ml' in self.recipe with new dose volumes (uL -> mL).
        Writes updated self.recipe back to self.recipe_path.
        """
        recipe = self.recipe

        # 1) Set *all existing* entries to zero (safety: prevent unintended aspirations)
        for chem in list(recipe.keys()):
            recipe[chem] = 0.0

        # 2) Apply suggestions (convert uL -> mL)
        total_ml = 0
        for name, ul in values.items():
            try:
                ul_f = float(ul)
            except Exception:
                log.error(f"{name}: non-numeric dose '{ul}' -> setting to 0")
                ul_f = 0.0

            ml = round(ul_f / 1000, 3)

            if name not in recipe:
                raise ValueError(f"{name} returned from Atinary not in YAML recipe_ml!")

            recipe[name] = ml

            total_ml += ml
            log.info(f"{name} dose volume updated to {ul_f}uL ({ml}mL).")

        log.info(f"New recipe total volume = {total_ml}ml.")

        if total_ml > self.electrolyte_volume:
            log.error(f"Total volume from Atinary is greater than expected electrolyte volume: {total_ml} > {self.electrolyte_volume}")

        # 3) Save back to YAML
        with open(self.recipe_path, "w", encoding="utf-8") as f:
            yaml.dump(self.recipe, f, sort_keys=False)

    def calculate_cost(self) -> float:
        """
        Compute total cost from self.cfg after update_yaml_volumes() has run.
        - Uses self.recipe['recipe_ml'] volumes in mL
        - Uses self.cfg['chemicals'][name]['cost'] as cost per mL
        """
        recipe = self.recipe
        chems  = self.cfg.get("chemicals", {})

        total = 0
        for name, vol_ml in recipe.items():
            meta = chems.get(name)
            if not meta or "cost" not in meta:
                log.warning(f"calculate_cost: missing chem/cost for '{name}', skipping.")
                continue
            try:
                v = float(vol_ml)
                c = float(meta["cost"])
            except (TypeError, ValueError):
                log.error(f"calculate_cost: non-numeric vol/cost for '{name}' (vol='{vol_ml}', cost='{meta.get('cost')}'), skipping.")
                continue
            total += v * c

        logging.info(f"Total recipe cost = {total:.6g}")
        return total

    def aggregate_results_from_ids(
        self,
        agg_column: str,
        ids: Optional[List[str]] = None,
        agg_fn: Optional[str] = "mean",
        mode: Optional[str] = "EIS"
    ) -> float:
        """
        Load results CSV and aggregate values for the given IDs.

        Args:
            ids: List of experiment IDs to filter by.
            agg_column: Column name in CSV to aggregate.
            agg_fn: Aggregation function ('mean', 'sum', 'median').

        Returns:
            Aggregated value (float) or None if no matching rows.
        """

        if mode == "CV" and not ids:
            ids = self.latest_cv_ids
        elif mode == "EIS" and not ids:
            ids = self.latest_eis_ids

        try:
            master_csv = os.path.join(self.cell.squid.results_path, f"{mode}_Results.csv")
            df = pd.read_csv(master_csv)
        except FileNotFoundError:
            logging.error(f"Results CSV not found: {self.cell.master_csv}")
            return None
        except Exception as e:
            logging.error(f"Error reading results CSV: {e}")
            return None

        # Filter for matching IDs
        df_filtered = df[df["Unique ID"].isin(ids)]

        if df_filtered.empty:
            logging.warning("No matching rows found for latest IDs.")
            return None

        if agg_column not in df_filtered.columns:
            logging.error(f"Column '{agg_column}' not found in results CSV.")
            return None

        try:
            result = df_filtered[agg_column].agg(agg_fn)
            logging.info(f"Aggregated {agg_fn} of '{agg_column}' for {len(df_filtered)} rows = {result}")
            return result
        
        except Exception as e:
            logging.error(f"Error aggregating column '{agg_column}': {e}")
            return None

    # -------------------- End Public API --------------------

    # -------------------- Internals --------------------
    def _load_config(self, path: str) -> dict:
        """
        Load YAML configuration file for hardware and experiment settings.
        
        Args:
            path (str): Path to YAML config file.
        Returns:
            dict: Parsed configuration dictionary.
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
        
    def _chem_exists(self, chemical: str) -> bool:
        if self.chem_map.get(chemical) is not None:
            return True
        else:
            return False

    def _where(self, chemical: str) -> Tuple[str, int]:
        """
        Get the pump controller and channel index for a given chemical.
        Validates mapping from config.
        
        Args:
            chemical (str): Chemical name.
        Returns:
            Tuple[str, int]: Controller ('A' or 'B') and pump index (1-4).
        """
        meta = self.chem_map[chemical]
        ctl = meta["controller"].strip().upper()
        idx = int(meta["pump_index"])

        if ctl not in ("A", "B") or not (1 <= idx <= 4):
            raise ValueError(f"Bad mapping for {chemical}: {meta}")
        
        return ctl, idx
    
    def _construct_mixture_title(self, recipe_ml: Dict[str, float]):
        """
        Construct a string title for the mixture based on chemical names and volumes.
        Used for metadata and logging.
        
        Args:
            recipe_ml (Dict[str, float]): Mapping of chemical names to volumes (ml).
        Returns:
            str: Mixture title string.
        """
        parts = [
            f"{name}:{amount}ml"
            for name, amount in recipe_ml.items()
            if amount > 0
        ]
        return "-".join(parts)

    def _transfer_pump(self, ctl: str, pump_index: int, volume_ml: float, pwm: float, check: bool) -> None:
        """
        Transfer a specified volume using a given pump controller and channel.
        Calculates pump time based on ml/s and PWM settings from config.
        
        Args:
            ctl (str): Controller ('A' or 'B').
            pump_index (int): Pump channel index (1-4).
            volume_ml (float): Volume to transfer in ml.
            check (bool): Whether to check hardware response after transfer.
        """
        pump = self.pumpA if ctl == "A" else self.pumpB
        mlps = self.cfg["pumps"].get("ml_per_s", 0.5)

        if mlps <= 0 or abs(pwm) > 100:
            raise ValueError("Incorrect variables given for PWM!")
        
        pump.transfer_pump(pump_no=pump_index, pwm=pwm, seconds=float(volume_ml / mlps), check=check)

    def _wait_for_responses(self):
        """
        Wait for both pump controllers to confirm completion of their last command.
        """
        self.pumpA.check_response()
        self.pumpB.check_response()

    def _get_pump(self, ctl: str):
        """
        Return correct pump controller instances based on if A or B passed.
        """
        ctl = ctl.upper().strip()
        if ctl == "A": 
            return self.pumpA
        if ctl == "B":
            return self.pumpB
        
        raise ValueError(f"Unknown controller '{ctl}'")
        
    def _single_dose(self, chemical: str, volume_ml: float, flow_rate: float) -> None:
        """
        Dose a single chemical using its mapped pump controller and channel.
        Used for priming and single chemical dosing.
        
        Args:
            chemical (str): Chemical name.
            volume_ml (float): Volume to dose in ml.
        """
        if not self._chem_exists(chemical):
            return
        
        ctl, idx = self._where(chemical)

        pump = self.pumpA if ctl == "A" else self.pumpB
        
        log.info(f"Dosing {chemical}: {volume_ml:.3f} ml on {ctl}[{idx}]")
        pump.single_pump(pump_no=idx, ml=volume_ml, flow_rate=flow_rate)
