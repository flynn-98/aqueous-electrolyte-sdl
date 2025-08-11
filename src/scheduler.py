from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import yaml

# Local hardware modules (from the user's codebase)
from pump_controller import pump_controller
from temperature_controller import peltier
from squidstat import squidstat
from test_cell import measurements

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class scheduler:
    def __init__(self, config_path: str, sim: bool = False) -> None:
        """
        Load configuration and create hardware instances.

        Args:
            config_path: path to YAML file (see example).
            sim: pass True to avoid talking to serial hardware during dry-runs.
        """
        self.sim = sim
        self.cfg = self._load_config(config_path)

        # --- Instantiate hardware ---
        # Pumps (two 4‑channel controllers → 8 chemicals total)
        pumpA_cfg = self.cfg["serial"]["pump_controller_A"]
        pumpB_cfg = self.cfg["serial"]["pump_controller_B"]
        self.pumpA = pump_controller(COM=pumpA_cfg["port"], sim=self.sim)
        self.pumpB = pump_controller(COM=pumpB_cfg["port"], sim=self.sim)

        # Temperature controller
        pel_cfg = self.cfg["serial"]["temperature_controller"]
        self.tec = peltier(COM=pel_cfg["port"], sim=self.sim)

        # Potentiostat
        squid_cfg = self.cfg["serial"]["squidstat"]
        self.squid = squidstat(port=squid_cfg.get("port", None), sim=self.sim)

        # Measurements helper (for data/paths, post-processing, etc.)
        meas_cfg = self.cfg.get("measurements", {})
        self.meas = measurements(
            squid=self.squid,
            results_root=meas_cfg.get("results_root", "./results"),
            sim=self.sim
        )

        # Map chemicals to (controller, pump_index)
        # Example entry in YAML:
        # chemicals:
        #   Na2SO4: { controller: A, pump_index: 0, prime_ml: 6.0 }
        self.chem_map = self.cfg["chemicals"]

        # Priming state per chemical
        self._primed = {name: False for name in self.chem_map.keys()}

    # -------------------- Public API --------------------

    def ensure_primed(self) -> None:
        """
        Prime all chemical lines with the configured prime volume (ml).
        Skips chemicals already primed in this process.
        """
        defaults = self.cfg["pumps"].get("defaults", {})
        default_pwm = float(defaults.get("pwm", 0.6))

        for chem, meta in self.chem_map.items():
            if self._primed.get(chem, False):
                continue
            prime_ml = float(meta.get("prime_ml", 0.0))
            if prime_ml <= 0:
                log.info(f"Skipping prime for {chem} (prime_ml <= 0).")
                self._primed[chem] = True
                continue
            self._dose(chem, volume_ml=prime_ml, pwm=default_pwm)
            log.info(f"Primed {chem} with {prime_ml} ml.")
            self._primed[chem] = True

    def make_mixture(self, recipe_ml: Dict[str, float]) -> None:
        """
        Create a mixture by dosing specified volumes (ml) of each chemical.
        Uses multi‑pump calls per controller for concurrency when possible.

        Args:
            recipe_ml: mapping chemical -> volume in ml
        """
        # Validate chemicals exist
        unknown = [k for k in recipe_ml.keys() if k not in self.chem_map]
        if unknown:
            raise ValueError(f"Unknown chemical(s): {unknown}")

        # Ensure lines are primed before first actual dosing
        if not all(self._primed.values()):
            log.info("Priming lines first...")
            self.ensure_primed()

        # Build per‑controller pump seconds arrays (len=4 each)
        secsA = [0.0, 0.0, 0.0, 0.0]
        secsB = [0.0, 0.0, 0.0, 0.0]
        pwmA = float(self.cfg["serial"]["pump_controller_A"].get("default_pwm", 0.6))
        pwmB = float(self.cfg["serial"]["pump_controller_B"].get("default_pwm", 0.6))

        for chem, vol_ml in recipe_ml.items():
            ctl, idx = self._where(chem)
            seconds = self._ml_to_seconds(ctl, idx, float(vol_ml))
            if ctl == "A":
                secsA[idx] = seconds
            else:
                secsB[idx] = seconds

        # Fire both controllers (start with controller A, then B)
        # Prefer the explicit methods exposed by provided pump_controller.py
        # - transfer_pump(pump_no, pwm, seconds)
        # - multi_pump(seconds_for_4_channels)   # if available in your file
        # - multiStepperPump(volumes)            # alt naming
        if any(s > 0 for s in secsA):
            self._multi_or_single(self.pumpA, secsA, pwmA)
        if any(s > 0 for s in secsB):
            self._multi_or_single(self.pumpB, secsB, pwmB)

    def run_temperature_sweep_with_eis(
        self,
        setpoints_C: List[float],
        soak_s: int = 120,
        eis_current_A: float = 0.0,
        freq_start_Hz: float = 10000.0,
        freq_stop_Hz: float = 0.5,
        points_per_decade: int = 8,
    ) -> None:
        """
        For each temperature setpoint: reach temperature, soak, run EIS (galvanostatic or OCP).
        Tweak to your preferred experiment builder (CV, OCP, EIS, etc.).
        """
        # Temperature controller sanity checks
        try:
            self.tec.handshake()
            self.tec.set_run_flag(True)
        except Exception as e:
            log.warning(f"Temperature controller handshake failed in sim={self.sim}: {e}")

        for T in setpoints_C:
            log.info(f"Setting temperature to {T:.1f} C")
            try:
                self.tec.set_temperature(T)
                self.tec.wait_until_temperature(
                    target_temperature=T,
                    tolerance=self.cfg["temperature"].get("tolerance_C", 0.3),
                    timeout=self.cfg["temperature"].get("timeout_s", 1800)
                )
            except Exception as e:
                log.warning(f"Temperature control step failed or skipped (sim?): {e}")

            # Soak
            log.info(f"Soaking for {soak_s} s at {T:.1f} C...")
            time.sleep(soak_s if not self.sim else 0.01)

            # Build and run the electrochemical experiment
            # Choose one pattern below that matches your squidstat.py; this shows both options.
            try:
                mode = "EIS_GALV"
                self.meas.update_data_space(mode)

                # Option A: explicit builder on squidstat
                if hasattr(self.squid, "build_EIS_galvanostatic_experiment"):
                    self.squid.build_EIS_galvanostatic_experiment(
                        current_A=eis_current_A,
                        start_frequency_Hz=freq_start_Hz,
                        end_frequency_Hz=freq_stop_Hz,
                        points_per_decade=points_per_decade
                    )
                    self.squid.run_experiment()

                # Option B: OCP/EIS helper (if you prefer OCP at each T)
                elif hasattr(self.squid, "build_OCP_experiment"):
                    self.squid.build_OCP_experiment(duration_s=soak_s)
                    self.squid.run_experiment()

                else:
                    log.error("No supported experiment builder found on squidstat.")

            except Exception as e:
                log.error(f"Electrochemical measurement failed: {e}")

        # Clear run flag on temperature controller
        try:
            self.tec.set_run_flag(False)
        except Exception:
            pass

    # -------------------- End Public API --------------------

    # -------------------- Internals --------------------
    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _where(self, chemical: str) -> Tuple[str, int]:
        """Return ('A' or 'B', pump_index 0..3) for a chemical."""
        meta = self.chem_map[chemical]
        ctl = meta["controller"].strip().upper()
        idx = int(meta["pump_index"])
        if ctl not in ("A", "B") or not (0 <= idx <= 3):
            raise ValueError(f"Bad mapping for {chemical}: {meta}")
        return ctl, idx

    def _ml_to_seconds(self, controller: str, pump_index: int, volume_ml: float) -> float:
        """
        Convert ml to seconds based on calibration in YAML.
        Expects ml_per_s per pump in config:
          serial:
            pump_controller_A:
              ml_per_s: [.. four floats ..]
        """
        key = "pump_controller_A" if controller == "A" else "pump_controller_B"
        mlps = self.cfg["serial"][key]["ml_per_s"][pump_index]
        mlps = float(mlps)
        if mlps <= 0:
            raise ValueError(f"ml_per_s for controller {controller} pump {pump_index} must be > 0")
        seconds = volume_ml / mlps
        return float(seconds)

    def _multi_or_single(self, pump: pump_controller, secs4: List[float], pwm: float) -> None:
        """
        Use the most capable method available on the provided pump_controller.
        Falls back to single transfers if multi method is absent.
        - Prefer: multi_pump([sec_ch0, sec_ch1, sec_ch2, sec_ch3], pwm=pwm)
        - Alt:   multiStepperPump([sec_ch0, sec_ch1, sec_ch2, sec_ch3])  # if firmware handles pwm
        - Fallback: transfer_pump(pump_no, pwm, seconds)
        """
        # Reduce floating noise
        secs4 = [float(s) for s in secs4]

        # Try common multi-channel variants
        if hasattr(pump, "multi_pump"):
            try:
                pump.multi_pump(secs4, pwm=pwm)  # type: ignore[arg-type]
                return
            except TypeError:
                # Some versions might be (secs4) only
                pump.multi_pump(secs4)  # type: ignore[call-arg]
                return
            except Exception as e:
                log.warning(f"multi_pump failed, falling back to singles: {e}")

        if hasattr(pump, "multiStepperPump"):
            try:
                pump.multiStepperPump(secs4)  # firmware embeds pwm
                return
            except Exception as e:
                log.warning(f"multiStepperPump failed, falling back to singles: {e}")

        # Fallback: drive channel by channel
        for i, seconds in enumerate(secs4):
            if seconds <= 0:
                continue
            try:
                pump.transfer_pump(pump_no=i, pwm=pwm, seconds=seconds)
            except Exception as e:
                log.error(f"transfer_pump failed for channel {i}: {e}")

    def _dose(self, chemical: str, volume_ml: float, pwm: float) -> None:
        ctl, idx = self._where(chemical)
        pump = self.pumpA if ctl == "A" else self.pumpB
        seconds = self._ml_to_seconds(ctl, idx, volume_ml)
        log.info(f"Dosing {chemical}: {volume_ml:.3f} ml → {seconds:.3f} s @ PWM {pwm:.2f} on {ctl}[{idx}]")
        pump.transfer_pump(pump_no=idx, pwm=pwm, seconds=seconds)


# -------------------- Convenience runner --------------------
def run_basic_experiment(config_path: str, recipe_ml: Dict[str, float]) -> None:
    """
    Example usage: load config, prime lines, make mixture, and run a temperature sweep with EIS.
    Adjust to exactly mirror the previous version's ordering/experiment types.
    """
    s = scheduler(config_path=config_path, sim=False)
    s.ensure_primed()
    s.make_mixture(recipe_ml)

    # Temperatures and EIS parameters come from YAML
    temps = s.cfg["temperature"]["setpoints_C"]
    soak = int(s.cfg["temperature"].get("soak_s", 120))
    eis = s.cfg.get("eis", {})
    s.run_temperature_sweep_with_eis(
        setpoints_C=temps,
        soak_s=soak,
        eis_current_A=float(eis.get("current_A", 0.0)),
        freq_start_Hz=float(eis.get("freq_start_Hz", 10000.0)),
        freq_stop_Hz=float(eis.get("freq_stop_Hz", 0.5)),
        points_per_decade=int(eis.get("ppd", 8)),
    )


if __name__ == "__main__":
    # Minimal smoke test (sim=True recommended for first run)
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--sim", action="store_true", help="Run without serial I/O")
    args = p.parse_args()

    sched = scheduler(args.config, sim=args.sim)
    # Example recipe: read from YAML "example_recipe_ml" if present
    example = sched.cfg.get("example_recipe_ml", {})
    if example:
        sched.ensure_primed()
        sched.make_mixture(example)
        temps = sched.cfg["temperature"]["setpoints_C"]
        sched.run_temperature_sweep_with_eis(setpoints_C=temps)
    else:
        log.info("No example_recipe_ml in config; nothing scheduled.")
