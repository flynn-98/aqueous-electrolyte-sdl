import logging
import os
from datetime import datetime
from csv import DictWriter
import time
from typing import Callable, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from impedance import preprocessing
from impedance.models.circuits import Randles
from impedance.visualization import plot_nyquist, plot_bode
from scipy.signal import find_peaks

from squidstat import squidstat

def skip_if_sim(default_return = None):
    def decorator(func):
        def wrapper(self: measurements, *args, **kwargs):
            if self.sim:
                return default_return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class measurements:
    def __init__(self, squid_port: str, instrument: str, results_path: str, channel: int = 0, squid_sim: bool = False) -> None:

        self.sim = squid_sim
        self.squid = squidstat(COM=squid_port, instrument=instrument, results_path=results_path, channel=channel, squid_sim=squid_sim)

        # Metadata
        self.user = "Unknown"
        self.electrolyte = "Unknown"
        self.project = "Unknown"

        # Default values (might be overwritten)
        self.conductivity_acceptance = 0.5 #mS/cm
        self.esw_acceptance = 0.5 #V
        self.max_attempts = 10

        self.epsilon_0 = 8.8541878128e-12 # vacuum permittivity
        self.cell_constant = 8 # to be set from hardcoded values
        self.test_cell_volume = 2.4 # ml

        self.master_csv = None

        self.eis_fieldnames = [
            'Unique ID',
            'User', 
            'Project', 
            'Electrolyte',
            'Cell Constant',
            'Target Temperature (C)',
            'Start Temperature (C)',
            'End Temperature (C)',
            'Start OCP (V)',
            'End OCP (V)',
            'Settle Time (mins)',
            'Settle Error (mS/cm)',
            'Minimum Ohmic Resistance (Ohms)',
            'Minimum Ionic Conductivity (mS/cm)',
            'Fitted Ionic Conductivity (mS/cm)',
            'Randles Ohmic Resistance (Ohms)',
            'Randles Ionic Conductivity (mS/cm)',
            'Randles nEVM',
            'Start Frequency (Hz)',
            'End Frequency (Hz)',
            'Points / Decade',
            'Voltage Amplitude (V)',
            'Voltage Bias (V)',
            'Timestamp',
        ]

        self.cv_fieldnames = [
            'Unique ID',
            'User', 
            'Project', 
            'Electrolyte',
            'Target Temperature (C)',
            'Start Temperature (C)',
            'End Temperature (C)',
            'Start OCP (V)',
            'End OCP (V)',
            'Settle Time (mins)',
            'Settle Error (V)',
            'ESW (V)',
            'Start Voltage (V)',
            'First Voltage Limit (V)',
            'Second Voltage Limit (V)',
            'End Voltage (V)',
            'Scan Rate (V/s)',
            'Sampling Interval (s)',
            'Cycles',
            'Timestamp',
        ]

    def get_indentifier(self) -> str:
        now = datetime.now()
        return f"ID_{now.strftime('%d-%m-%Y_%H-%M-%S')}_{self.user}_{self.project}_"
    
    def update_experiment_dir(self, mode: str) -> None:
        if os.path.basename(self.squid.results_path) == "results":
            self.squid.results_path = os.path.join(self.squid.results_path, mode)

        elif os.path.basename(self.squid.results_path) != mode:
            self.squid.results_path = os.path.join(os.path.dirname(self.squid.results_path), mode)

        # Create folder if not already
        if not os.path.exists(self.squid.results_path):
            os.mkdir(self.squid.results_path)
    
    def update_data_space(self, mode: str) -> str:
        # Adjust squidstat results path to point raw data towards correct folder (if not done already)
        self.update_experiment_dir(mode)
        
        self.master_csv = os.path.join(self.squid.results_path, f"{mode}_Results.csv")
        # Fix to prevent repeated EIS addition

        # Create folder if not already
        if not os.path.exists(self.squid.results_path):
            os.mkdir(self.squid.results_path)

        # Create file with header if first time running code on PC
        if not os.path.exists(self.master_csv):
            logging.info("Creating CSV master file..")
            with open(self.master_csv, 'a+', newline='') as file:
                writer = DictWriter(file, fieldnames=self.eis_fieldnames)
                writer.writeheader()

        return self.get_indentifier() + mode
    
    def metadata_check(self):
        if self.user == "Unknown" or self.project == "Unknown" or self.electrolyte == "Unknown":
            raise RuntimeError("Metadata incomplete!")
    
    @skip_if_sim()
    def perform_EIS_experiment(self,
        start_frequency: float,
        end_frequency: float,
        points_per_decade: int,
        voltage_amplitude: float,
        voltage_bias: float,
        target_temperature: float,
        get_temperature_fn: Optional[Callable[[], float]] = None,
        measurements: int = 1
        ) -> None:
        
        self.metadata_check()

        # Update dir, csv and id for robust data collection
        id = self.update_data_space("EIS")

        for m in range(measurements):

            start_time = time.time()

            if get_temperature_fn:
                start_temp = get_temperature_fn()
            else:
                start_temp = "NA"

            # Perform initial OCP test
            self.squid.take_quick_measurements("OCP_START", 1)
            start_ocp = self.get_avg_working_voltage("OCP_START")
    
            count = 0
            error = self.conductivity_acceptance + 1
            last_result = 0

            # Loop until stable values (stability checked using minimum bulk conductivity)
            while (count < self.max_attempts and error > self.conductivity_acceptance):
                logging.info(f"Repeating until conductivity is stable (measurement #{count+1})..")
                time.sleep(10)

                count_id = id + f"_{count+1}"
                end_time = time.time()

                self.squid.build_EIS_potentiostatic_experiment(
                    start_frequency=start_frequency,
                    end_frequency=end_frequency,
                    points_per_decade=points_per_decade,
                    voltage_amplitude=voltage_amplitude,
                    voltage_bias=voltage_bias
                )

                # First measurements
                self.squid.run_experiment()
                self.squid.save_data(count_id)

                if get_temperature_fn:
                    end_temp = get_temperature_fn()
                else:
                    end_temp = "NA"

                # Get data
                metadata = {
                    "temperature": f"{(start_temp)}C",
                    "start_frequency": f"{start_frequency}Hz",
                    "end_frequency": f"{end_frequency}Hz",
                    "points_per_decade": f"{points_per_decade}",
                    "voltage_amplitude": f"{voltage_amplitude}V",
                    "voltage_bias": f"{voltage_bias}V"
                }
                results = self.get_impedance_properties(identifier=count_id, metadata=metadata)

                if count > 0:
                    error = abs(results[1] - last_result)
                    logging.info(f"Difference in last two conductivity estimates: {error}mS/cm")

                # Leave loop if second measurement
                if m > 0:
                    break
                else:
                    last_result = results[1]
                    count += 1

            # Perform final OCP test
            self.squid.take_quick_measurements("OCP_END", 1)
            end_ocp = self.get_avg_working_voltage("OCP_END")

            # Write to CSV
            with open(self.master_csv, 'a', newline='') as file:
                writer = DictWriter(file, fieldnames=self.eis_fieldnames)
                writer.writerow({
                    'Unique ID': id, 
                    'User': self.user, 
                    'Project': self.project, 
                    'Electrolyte': self.electrolyte,
                    'Cell Constant': self.cell_constant,
                    'Target Temperature (C)': target_temperature,
                    'Start Temperature (C)': start_temp,
                    'End Temperature (C)': end_temp,
                    'Start OCP (V)': start_ocp,
                    'End OCP (V)': end_ocp,
                    'Settle Time (mins)': round( (end_time - start_time) / 60, 2),
                    'Settle Error (mS/cm)': error,
                    'Minimum Ohmic Resistance (Ohms)': results[0],
                    'Minimum Ionic Conductivity (mS/cm)': results[1],
                    'Fitted Ionic Conductivity (mS/cm)': results[2],
                    'Randles Ohmic Resistance (Ohms)': results[3],
                    'Randles Ionic Conductivity (mS/cm)': results[4],
                    'Randles nEVM': results[5],
                    'Start Frequency (Hz)': start_frequency,
                    'End Frequency (Hz)': end_frequency,
                    'Points / Decade': points_per_decade,
                    'Voltage Amplitude (V)': voltage_amplitude,
                    'Voltage Bias (V)': voltage_bias,
                    'Timestamp': datetime.now(),
                    })
            
    def plot_EIS(self, f: any, Z: any, Z_fit: any, identifier: str = "na", metadata: dict = {}) -> list:
        logging.info("Saving EIS plot (Dataset " + identifier + ")..")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))  # side-by-side layout

        # Nyquist plot
        plot_nyquist(Z, fmt='o', scale=10, ax=ax1)
        plot_nyquist(Z_fit, fmt='-', scale=10, ax=ax1)
        ax1.set_title("Nyquist Plot")
        ax1.legend(['Real Data', 'Model'])
        ax1.set_aspect('equal', adjustable='datalim')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)

        # Bode plot
        plot_bode(f, Z, fmt='o', scale=10, axes=[ax2, ax3])
        plot_bode(f, Z_fit, fmt='-', scale=10, axes=[ax2, ax3])
        ax2.set_title("Bode Plot")
        ax2.legend(['Real Data', 'Model'])
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)

        # Titles
        fig.suptitle(
            f"{self.project} – EIS Analysis: {identifier}",
            fontsize=16, fontweight='bold', y=1.0
        )
        fig.text(
            0.5, 0.975,
            f"User: {self.user} | Electrolyte: {self.electrolyte}",
            ha='center', fontsize=11, style='italic'
        )

        # Display passed metadata
        meta_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        fig.text(
            1.02, 0.5, meta_text, va='center', ha='left', fontsize=10
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.squid.results_path, identifier + ".png"), dpi=300)
        plt.close()

    def get_impedance_properties(self, identifier: str = "na", metadata: dict = {}, plot: bool = True) -> float:    
        # AC data required for impedance properties
        data = pd.read_csv(self.squid.get_ac_path(identifier)).to_numpy()

        frequency = data[:, 0]
        z_real = data[:, 1]

        z_img = -data[:, 2]
        abs_imag = np.abs(data[:, 2])
        
        # Finding ohmic resistance
        min_imag = np.amin(abs_imag) # minimum of abs(Imag)

        min_index = np.where(abs_imag == min_imag)[0][0] # take the first index if multiple instances
        min_ohmic_resistance = np.round(z_real[min_index], 5).item() # take real value corresponding to minimum imag

        logging.info(f"Minimum Ohmic Resistance calculated as {min_ohmic_resistance}Ohms.")

        min_ionic_conductivity = 1000 * self.cell_constant / min_ohmic_resistance

        logging.info(f"Minimum Ionic Conductivity calculated as {min_ionic_conductivity}mS/cm.")

        # Finding Ionic conductivity
        conductivity = np.empty((1,0))
        tan_delta = np.empty((1,0))

        for z_r, z_i, f in zip(z_real, z_img, frequency):
            z_square = z_r ** 2 + z_i ** 2
            epsilon_real = z_i * self.cell_constant / (2 * np.pi * f * self.epsilon_0 * z_square)
            epsilon_img = z_r * self.cell_constant / (2 * np.pi * f * self.epsilon_0 * z_square)

            conductivity = np.append(conductivity, np.array([self.epsilon_0 * epsilon_img * 2 * np.pi * f]))
            tan_delta = np.append(tan_delta, np.array([epsilon_img / epsilon_real]))

        max_index = np.argmax(tan_delta)
        fitted_ionic_conductivity = np.round(conductivity[max_index], 5).item() * 1000 # ms/cm

        logging.info(f"Fitted Ionic Conductivity calculated as {fitted_ionic_conductivity}mS/cm.")

        initial_guess = [100, 0.005, 0.001, 200, 0.1, 0.9]
        circuit = Randles(initial_guess=initial_guess, CPE=True)
        frequencies, Z = preprocessing.readCSV(self.squid.get_ac_path(identifier))
        frequencies, Z = preprocessing.ignoreBelowX(frequencies, Z)

        circuit.fit(frequencies, Z)

        randles_ohmic_resistance = circuit.parameters_[0]
        randles_ionic_conductivity = 1000 * self.cell_constant / randles_ohmic_resistance

        logging.info(f"Randles Ohmic Resistance calculated as {randles_ohmic_resistance}Ohms.")
        logging.info(f"Randles Ionic Conductivity calculated as {randles_ionic_conductivity}mS/cm.")

        Z_fit = circuit.predict(frequencies)

        residuals = Z - Z_fit
        evm = np.linalg.norm(residuals) / np.linalg.norm(Z) # <0.1 is good fit, >0.3 bad fit

        logging.info(f"Randles fit nEVM score: {evm} (good fit < 0.1, bad fit > 0.3).")

        if plot is True:
            self.plot_EIS(frequencies, Z, Z_fit, identifier, metadata)

        return [round(min_ohmic_resistance, 3), round(min_ionic_conductivity, 3), round(fitted_ionic_conductivity, 3), round(randles_ohmic_resistance, 3), round(randles_ionic_conductivity, 3), round(evm, 4)]
    
    def get_avg_working_voltage(self, identifier: str = "na") -> float:
        data = pd.read_csv(self.squid.get_dc_path(identifier)).to_numpy()

        return np.average(data[:, 1])
    
    @skip_if_sim()
    def perform_CV_experiment(self,
        first_voltage_limit: float,
        second_voltage_limit: float,
        scan_rate: float,
        sampling_interval: float,
        cycles: int,
        target_temperature: float,
        get_temperature_fn: Optional[Callable[[], float]] = None,
        measurements: int = 1
        ) -> None:

        self.metadata_check()

        # Update dir, csv and id for robust data collection
        id = self.update_data_space("CV")

        for m in range(measurements):

            start_time = time.time()

            if get_temperature_fn:
                start_temp = get_temperature_fn()
            else:
                start_temp = "NA"

            # Perform initial OCP test
            self.squid.take_quick_measurements("OCP_START", 1)
            start_ocp = self.get_avg_working_voltage("OCP_START")

            start_voltage = start_ocp
            end_voltage = start_voltage
    
            count = 0
            error = self.esw_acceptance + 1
            last_result = 0

            # Loop until stable values (stability checked using minimum bulk conductivity)
            while (count < self.max_attempts and error > self.esw_acceptance):
                logging.info(f"Repeating until ESW is stable (measurement #{count+1})..")
                time.sleep(10)

                end_time = time.time()
                count_id = id + f"_{count+1}"

                self.squid.build_cyclic_voltammetry_experiment(
                    start_voltage=start_voltage,
                    first_voltage_limit=first_voltage_limit,
                    second_voltage_limit=second_voltage_limit,
                    end_voltage=end_voltage,
                    scan_rate=scan_rate,
                    sampling_interval=sampling_interval,
                    cycles=cycles
                    )
                
                # First measurements
                self.squid.run_experiment()
                self.squid.save_data(count_id)

                if get_temperature_fn:
                    end_temp = get_temperature_fn()
                else:
                    end_temp = "NA"

                # Get data TO REPLACE
                metadata = {
                    "temperature": f"{start_temp}C",
                    "first_voltage_limit": f"{first_voltage_limit}V",
                    "second_voltage_limit": f"{second_voltage_limit}V",
                    "scan_rate": f"{scan_rate}V/s",
                    "sampling_interval": f"{sampling_interval}s",
                    "cycles": f"{cycles}"
                }
                result = self.get_electrochemical_stability_window(identifier=count_id, metadata=metadata)

                if count > 0:
                    error = abs(result - last_result)
                    logging.info(f"Difference in last two ESW estimates: {error}V")

                # Leave loop if second measurement
                if m > 0:
                    break
                else:
                    last_result = result
                    count += 1

            # Perform final OCP test
            self.squid.take_quick_measurements("OCP_END", 1)
            end_ocp = self.get_avg_working_voltage("OCP_END")

            # Write to CSV
            with open(self.master_csv, 'a', newline='') as file:
                writer = DictWriter(file, fieldnames=self.cv_fieldnames)
                writer.writerow({
                    'Unique ID': id, 
                    'User': self.user, 
                    'Project': self.project, 
                    'Electrolyte': self.electrolyte,
                    'Target Temperature (C)': target_temperature,
                    'Start Temperature (C)': start_temp,
                    'End Temperature (C)': end_temp,
                    'Start OCP (V)': start_ocp,
                    'End OCP (V)': end_ocp,
                    'Settle Time (mins)': round( (end_time - start_time) / 60, 2),
                    'Settle Error (V)': error,
                    'ESW (V)': result,
                    'Start Voltage (V)': start_voltage,
                    'First Voltage Limit (V)': first_voltage_limit,
                    'Second Voltage Limit (V)': second_voltage_limit,
                    'End Voltage (V)': end_voltage,
                    'Scan Rate (V/s)': scan_rate,
                    'Sampling Interval (s)': sampling_interval,
                    'Cycles': cycles,
                    })

    def plot_cyclic_voltammogram(self, voltage, current, identifier: str = "na", metadata: dict = {}) -> None:
        logging.info(f"Saving CV plot (Dataset {identifier})...")

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(voltage, current*1e6, linewidth=1.5)
        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("Current [uA]")
        ax.set_title("Cyclic Voltammogram")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Metadata and identifiers
        fig.suptitle(
            f"{self.project} – CV Analysis: {identifier}",
            fontsize=16, fontweight='bold', y=0.98
        )
        fig.text(
            0.5, 0.94,
            f"User: {self.user} | Electrolyte: {self.electrolyte}",
            ha='center', fontsize=11, style='italic'
        )

        # Display passed metadata
        meta_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
        fig.text(
            1.02, 0.5, meta_text, va='center', ha='left', fontsize=10, transform=ax.transAxes
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(self.squid.results_path, identifier + ".png"), dpi=300)
        plt.close()

    def get_electrochemical_stability_window(self, identifier: str = "na", metadata: dict = {}, plot: bool = True) -> float:
        data = pd.read_csv(self.squid.get_dc_path(identifier))

        voltage = data["Working Electrode Voltage [V]"].to_numpy()
        current = data["Working Electrode Current [A]"].to_numpy()

        # Detect anodic (oxidation) peak (max current)
        anodic_peaks, _ = find_peaks(current, prominence=1e-6)
        cathodic_peaks, _ = find_peaks(-current, prominence=1e-6)

        if plot:
            self.plot_cyclic_voltammogram(voltage, current, identifier, metadata)

        if len(anodic_peaks) == 0 or len(cathodic_peaks) == 0:
            logging.warning("Insufficient data to determine ESW — peaks not found.")
            return 0.0

        anodic_voltage = voltage[anodic_peaks[np.argmax(current[anodic_peaks])]]
        cathodic_voltage = voltage[cathodic_peaks[np.argmax(-current[cathodic_peaks])]]

        esw = round(abs(anodic_voltage - cathodic_voltage), 3)

        logging.info(f"Anodic peak at {anodic_voltage:.3f}V, Cathodic peak at {cathodic_voltage:.3f}V, ESW = {esw}V")

        return esw