import logging
import time
from typing import Dict, List, Tuple, Optional

import yaml

# Local hardware modules
from src.pump_controller import pump_controller
from src.temperature_controller import peltier
from src.electrochem_system import measurements

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    filename="optimiser.log",
    filemode="a",
)

class scheduler:
    def __init__(self, config_path: str) -> None:
        """
        Initialize the scheduler for hardware control.
        Loads configuration from YAML, sets up pump controllers (A and B), temperature controller,
        potentiostat, and measurement system. Maps chemicals to pumps and sets priming state.
        
        Args:
            config_path (str): Path to YAML configuration file.
        """
        self.cfg = self._load_config(config_path)

        # Pumps (two 4‑channel controllers → 8 chemicals total)
        pumpA_cfg = self.cfg["serial"]["pump_controller_A"]
        pumpB_cfg = self.cfg["serial"]["pump_controller_B"]
        self.pumpA = pump_controller(COM=pumpA_cfg["port"], baud=pumpA_cfg["baud"], sim=pumpA_cfg["mock"])
        self.pumpB = pump_controller(COM=pumpB_cfg["port"], baud=pumpB_cfg["baud"], sim=pumpB_cfg["mock"])

        # Temperature controller
        pel_cfg = self.cfg["serial"]["temperature_controller"]
        self.tec = peltier(COM=pel_cfg["port"], baud=pel_cfg["baud"], sim=pel_cfg["mock"])

        # Potentiostat
        squid_cfg = self.cfg["serial"]["squidstat"]

        # Measurements helper (for data/paths, post-processing, etc.)
        meas_cfg = self.cfg.get("measurements", {})
        self.cell = measurements(
            squid_port=squid_cfg["port"], 
            instrument=squid_cfg["instrument"],
            results_path=meas_cfg.get("results_root", "./results") ,
            channel=squid_cfg["channel"],
            squid_sim=squid_cfg["mock"]
            )
        
        # Populate metadata
        self.cell.user = meas_cfg.get("user", "Unknown")
        self.cell.project = meas_cfg.get("project", "Unknown")
        self.cell.electrolyte = meas_cfg.get("electrolyte", "Unknown")
        self.cell.cell_constant = meas_cfg["cell_constant"]
        self.cell.test_cell_volume = meas_cfg["cell_volume"]

        # Populate temperature related constants
        temp_cfg = self.cfg.get("temperature", {})
        self.tec.allowable_error = temp_cfg["tolerance_C"]
        self.tec.steady_state = temp_cfg["steady_s"]
        self.tec.timeout = temp_cfg["timeout_s"]
        self.tec.equilibrium_time = temp_cfg["wait_s"]
        self.cell.max_attempts = temp_cfg["max_attempts"]

        # Map chemicals to (controller, pump_index)
        # Example entry in YAML:
        # chemicals:
        #   Na2SO4: { controller: A, pump_index: 0, prime_ml: 6.0 }
        self.chem_map = self.cfg["chemicals"]

        # Priming state per chemical (needed?)
        self._primed = {name: False for name in self.chem_map.keys()}

    # -------------------- Public API --------------------

    def ensure_primed(self) -> None:
        """
        Ensure all chemical lines are primed before dosing.
        For each chemical, if not already primed, dose the prime_ml volume, transfer to cell and waste,
        and wait for hardware responses. Updates priming state.
        """
        for chem, meta in self.chem_map.items():
            if self._primed.get(chem, False):
                continue

            prime_ml = meta.get("prime_ml", 2.0)

            if prime_ml <= 0 or prime_ml > 6:
                log.warning(f"Check prime volume for {chem}, skipping for now..")
                continue

            self._single_dose(chem, volume_ml=prime_ml)
            log.info(f"Primed {chem} with {prime_ml}ml.")

            self._primed[chem] = True

            self.transfer_to_cell(check=False)
            self.transfer_to_waste(check=False)

            self._wait_for_responses()

    def deprime_lines(self) -> None:
        """
        Remove priming solution from all chemical lines by transferring to waste.
        Useful for cleaning or resetting the system before/after experiments.
        """
        for chem, meta in self.chem_map.items():

            prime_ml = meta.get("prime_ml", 0.0)
            if prime_ml <= 0 or prime_ml > 6:
                log.warning(f"Check prime volume for {chem}, skipping for now..")
                continue

            self._single_dose(chem, volume_ml=-prime_ml)

            log.info(f"Deprimed {chem} with {-prime_ml}ml.")
            self._primed[chem] = False
    
    def make_mixture(self, recipe_ml: Dict[str, float]) -> None:
        """
        Dose a mixture of chemicals according to the provided recipe.
        Validates chemicals, updates electrolyte name, primes lines if needed, and doses each chemical
        using the correct pump controller and channel. Waits for hardware responses after dosing.
        
        Args:
            recipe_ml (Dict[str, float]): Mapping of chemical names to volumes (ml) to dose.
        """
        # Validate chemicals exist
        unknown = [k for k in recipe_ml.keys() if k not in self.chem_map]
        if unknown:
            raise ValueError(f"Unknown chemical(s): {unknown}")
        
        # Update electrolyte name using mixture
        self.cell.electrolyte = self._construct_mixture_title(recipe_ml)

        # Ensure lines are primed before first actual dosing
        if not all(self._primed.values()):
            log.info("Priming lines first..")
            self.ensure_primed()

        # Build per‑controller pump seconds arrays (len=4 each)
        ml_A = [0.0, 0.0, 0.0, 0.0]
        ml_B = [0.0, 0.0, 0.0, 0.0]

        for chem, vol_ml in recipe_ml.items():
            ctl, idx = self._where(chem)
            if ctl == "A":
                ml_A[idx-1] = vol_ml
            else:
                ml_B[idx-1] = vol_ml

        # Fire both controllers (start with controller A, then B)
        # First with check=False to fire both simulatenously, then check afterwards
        log.info("Mixing all chemicals simulateously..")
        log.info(f"Controller A: {ml_A}")
        log.info(f"Controller B: {ml_B}")

        self.pumpA.multi_pump(ml_A, check=False)
        self.pumpB.multi_pump(ml_B, check=False)
        self._wait_for_responses()

    def transfer_to_cell(self, check: bool = True):
        """
        Transfer the mixed solution from the mixing chamber to the test cell.
        Uses pump controller A, channel 1, and adds any extra volume specified in config.
        
        Args:
            check (bool): Whether to check hardware response after transfer.
        """
        log.info(f"Transferring {self.cell.test_cell_volume}ml to cell..")
        extra_vol = self.cfg["volumes"].get("mix_to_cell_ml", 0)

        self._transfer_pump("A", 1, self.cell.test_cell_volume + extra_vol, check)

    def transfer_to_waste(self, check: bool = True):
        """
        Transfer the solution from the test cell to waste.
        Uses pump controller B, configurable waste channel, and adds any extra volume specified in config.
        
        Args:
            check (bool): Whether to check hardware response after transfer.
        """
        extra_vol = self.cfg["volumes"].get("cell_to_waste_ml", 0)
        waste_no = self.cfg["volumes"].get("waste_no", 1)

        log.info(f"Transferring {self.cell.test_cell_volume}ml to waste #{waste_no}..")

        self._transfer_pump("B", waste_no, self.cell.test_cell_volume + extra_vol, check)

    def system_flush(self, cleaning_agent: str = "Ethanol", flushing_agent: str = "Milli-Q"):
        flush_volume = self.cfg["volumes"].get("flush_ml", 0)
        cleaning_time = self.cfg["temperature"].get("cleaning_delay_s", 60)

        log.info("Beginning heated cleaning procedure..")

        # Heated cleaning with agent
        self.tec.set_temperature(self.tec.max_temp)

        # Quick flush to clear any salt from lines
        log.info(f"Rinsing with {flushing_agent}.")
        self._single_dose(flushing_agent, flush_volume)
        self.transfer_to_cell(check=False)
        self.transfer_to_waste(check=False)
        self._wait_for_responses()

        log.info(f"Cleaning with {cleaning_agent}.")
        self._single_dose(cleaning_agent, flush_volume)
        self.transfer_to_cell()
        log.info(f"Cleaning for {cleaning_time}s.")
        time.sleep(cleaning_time)
        self.transfer_to_waste()

        # Final flush
        log.info(f"Final rinsing with {flushing_agent}.")
        self._single_dose(flushing_agent, flush_volume)
        self.transfer_to_cell(check=False)
        self.transfer_to_waste(check=False)
        self._wait_for_responses()

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
        
        for T in setpoints_C:
            log.info(f"Waiting until temperature = {T:.1f} C")
            if not self.tec.wait_until_temperature(T):
                raise RuntimeError("Temperature regulation failed!")

            # Build and run the electrochemical experiment
            try:
                self.cell.perform_EIS_experiment(
                    start_frequency = freq_start_Hz,
                    end_frequency = freq_stop_Hz,
                    points_per_decade = points_per_decade,
                    voltage_amplitude = voltage_amplitude,
                    voltage_bias = voltage_bias,
                    target_temperature = T,
                    get_temperature_fn = self.tec.get_t1_value,
                    measurements = measurements
                )

            except Exception as e:
                log.error(f"Electrochemical measurement failed: {e}")

        # Clear run flag on temperature controller
        self.tec.clear_run_flag()

    def run_basic_experiment(self, recipe_ml: Optional[Dict[str, float]] = None, deprime = False) -> None:
        """
        Run a basic experiment using the scheduler and hardware modules.
        Loads configuration, performs sanity checks, primes and doses chemicals, regulates temperature,
        and runs EIS measurements. Transfers solution to cell and waste as needed. Optionally deprime lines after experiment.
        
        Args:
            recipe_ml (Optional[Dict[str, float]]): Optional mapping of chemical names to volumes (ml) to dose.
            deprime (bool): If True, deprime lines after experiment.
        """
        start = time.time()

        # Temperatures and EIS parameters come from YAML
        temps = self.cfg["temperature"]["setpoints_C"]
        eis = self.cfg.get("eis", {})

        if not recipe_ml:
            recipe_ml = self.cfg.get("default_recipe_ml", {})

        # Sanity checks
        self.tec.handshake()
        self.cell.metadata_check()

        # Begin temperature regulation
        self.tec.set_temperature(temps[0])
        
        self.make_mixture(recipe_ml)

        self.transfer_to_cell()

        self.run_temperature_sweep_with_eis(
            setpoints_C = temps,
            freq_start_Hz = eis["freq_start_Hz"],
            freq_stop_Hz = eis["freq_end_Hz"],
            voltage_amplitude = eis["amplitude_v"],
            voltage_bias = eis["bias_v"],
            points_per_decade = eis["ppd"],
            measurements = eis["measurements_per_temp"],
        )

        self.transfer_to_waste()

        end = time.time() - start
        log.info(f"[TIMER] Experiment completed in {round(end / 60, 2)}mins.")

        if deprime:
            self.deprime_lines()

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

    def _transfer_pump(self, ctl: str, pump_index: int, volume_ml: float, check: bool) -> None:
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

        mlps = self.cfg["pumps"].get("ml_per_s", 1)
        pwm = self.cfg["pumps"].get("default_pwm", 60)

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
        
    def _single_dose(self, chemical: str, volume_ml: float) -> None:
        """
        Dose a single chemical using its mapped pump controller and channel.
        Used for priming and single chemical dosing.
        
        Args:
            chemical (str): Chemical name.
            volume_ml (float): Volume to dose in ml.
        """
        ctl, idx = self._where(chemical)

        pump = self.pumpA if ctl == "A" else self.pumpB
        
        log.info(f"Dosing {chemical}: {volume_ml:.3f} ml on {ctl}[{idx}]")
        pump.single_pump(pump_no=idx, volume=volume_ml)
