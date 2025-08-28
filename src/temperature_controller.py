import logging
import sys
import time
from math import floor
from typing import Callable

import matplotlib.pyplot as plt
import serial

logging.basicConfig(level = logging.INFO)

def skip_if_sim(default_return = None):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.sim:
                return default_return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

# To communicate with: https://lairdthermal.com/products/product-temperature-controllers/tc-xx-pr-59-temperature-controller

class PeltierModule:
    def __init__(self, COM: str, baud: int = 115200, sim: bool = False) -> None:
        self.sim = sim

        # Default temperature wait parameters (might be overwritten)
        self.allowable_error = 0.25 #C
        self.steady_state = 90 #s
        self.timeout = 1800 #s (30mins)

        # Fixed values
        self.max_temp = 40 #C
        self.min_temp = -20 #C

        self.input_voltage = 12.0 #V
        self.max_current = 10.0 #A
        self.min_current = 0.1 #A

        self.fan_current = 2.0 #A
        self.fan_voltage = 12.0

        # Thermisistor Steinhart coefficients NTC1 (testing only)
        self.A_coeff_1 = 1.396917e-3
        self.B_coeff_1 = 2.378257e-4
        self.C_coeff_1 = 9.372652e-8

        # Thermisistor Steinhart coefficients NTC2 (standard type)
        self.A_coeff_2 = 1.0373e-3
        self.B_coeff_2 = 2.3317e-4
        self.C_coeff_2 = 8.3896e-8

        # Heating/Cooling control
        self.heating_tc = 60 #%
        self.heating_Kp = 14
        self.heating_Ki = 0.2
        self.heating_Kd = 0.0

        self.cooling_tc = 85 #%
        self.cooling_Kp = 14
        self.cooling_Ki = 0.2
        self.cooling_Kd = 0.0

        self.subzero_tc = 100 #%
        self.subzero_Kp = 16
        self.subzero_Ki = 0.4
        self.subzero_Kd = 0.0

        self.run_flag = False

        self.temp_threshold = 15 #C, to set heating or cooling parameters
        self.subzero_threshold = 0 #C
        self.dead_band = 2 #+-% to prevent rapid switching

        if self.sim:
            logging.info("Simulated connection to temperature controller established.")

        else:
            logging.info("Configuring temperature controller serial port..")
            self.ser = serial.Serial(COM) 
            self.ser.baudrate = baud
            self.ser.bytesize = 8
            self.ser.parity = 'N' # No parity
            self.ser.stopbits = 1

            logging.info("Attempting to open temperature controller serial port..")

            if self.ser.isOpen() is False:
                self.ser.open()
            
        self.handshake()

        if self.set_regulator_mode() is True:
            logging.info("Temperature regulator PID mode successfully configured.")
        else:
            logging.error("Temperature regulator configuration failed.")
            sys.exit()

        if (self.set_tc_dead_band() is True):
            logging.info("Temperature regulator dead band settings successfully configured.")
        else:
            logging.error("Temperature regulator dead band configuration failed.")
            sys.exit()

        if self.set_voltage_alarm_settings() is True:
            logging.info("Temperature regulator voltage alarm settings successfully configured.")
        else:
            logging.error("Temperature regulator voltage alarm configuration failed.")
            sys.exit()

        if self.set_current_alarm_settings() is True:
            logging.info("Temperature regulator current alarm settings successfully configured.")
        else:
            logging.error("Temperature regulator current alarm configuration failed.")
            sys.exit()

        if self.configure_main_sensor() is True:
            logging.info("Temperature Sensor #1 successfully configured.")
        else:
            logging.error("Temperature Sensor #1 configuration failed.")
            sys.exit()

        if self.configure_heat_sink_sensor() is True:
            logging.info("Temperature sensor #2 successfully configured.")
        else:
            logging.error("Temperature sensor #2 configuration failed.")
            sys.exit()

        if self.set_main_steinhart_coeffs() is True:
            logging.info("Successfully updated steinhart coefficients for temperature sensor #1.")
        else:
            logging.error("Failed to update steinhart coefficients for temperature sensor #1.")
            sys.exit() 

        if self.set_heat_sink_steinhart_coeffs() is True:
            logging.info("Successfully updated steinhart coefficients for temperature sensor #2.")
        else:
            logging.error("Failed to update steinhart coefficients for temperature sensor #2.")
            sys.exit()         

        self.set_fan_modes()
        self.assess_status()      

    @skip_if_sim()
    def close_ser(self) -> None:
        if self.ser.isOpen():
            self.ser.close()

    @skip_if_sim(default_return="NA")
    def get_data(self) -> str:
        while self.ser.in_waiting == 0:
            pass

        return self.ser.readline().decode().rstrip()
    
    @skip_if_sim()
    def handshake(self) -> None:
        msg = "$LI"

        # returns '18245 TC-XX-PR-59 REV2.6'

        self.ser.write((msg+'\r').encode('ascii'))

        repeat = self.get_data()
        info = self.get_data()

        split = info.split(" ")[1]

        if split == "TC-XX-PR-59" and repeat == msg:
            logging.info("TEC located: " + info)
        else:
            raise RuntimeError(f"Temperature controller handshake failed: check connection! (Response: {split})")

    @skip_if_sim()
    def set_run_flag(self) -> None:
        msg = "$W"

        if not self.run_flag:
            self.ser.write((msg+'\r').encode('ascii'))
            repeat = self.get_data().split(" ")[1]

            if self.get_data() == "Run" and repeat == msg:
                logging.info("Temperature controller Run flag set.")
                self.run_flag = True
            else:
                logging.error("Failed to set temperature controller Run flag.")

    @skip_if_sim()
    def clear_run_flag(self) -> None:
        msg = "$Q"

        if self.run_flag:
            self.ser.write((msg+'\r').encode('ascii'))
            repeat = self.get_data().split(" ")[1]

            if self.get_data() == "Stop" and repeat == msg:
                logging.info("Temperature controller Run flag cleared.")
                self.run_flag = False
            else:   
                logging.error("Failed to clear temperature controller Run flag.")

    @skip_if_sim(default_return="0000 0000 0000")
    def get_status(self) -> str:
        msg = "$S"

        self.ser.write((msg+'\r').encode('ascii'))
        repeat = self.get_data().split(" ")[1]

        if repeat == msg:
            logging.info("Temperature controller status received.")
        else:   
            logging.error("Failed to get temperature controller status.")

        return self.get_data()
    
    @skip_if_sim(default_return="0000 0000 0000")
    def clear_status(self) -> str:
        msg = "$SC"

        self.ser.write((msg+'\r').encode('ascii'))
        repeat = self.get_data().split(" ")[1]

        if repeat == msg:
            logging.info("Temperature controller status cleared.")
        else:   
            logging.error("Failed to clear temperature controller status.")

        return self.get_data()

    @skip_if_sim()
    def assess_status(self) -> None:
        status = self.get_status()

        if status != "0000 0000 0000":
            logging.error("Status Error: " + status)

            status = self.clear_status()
            logging.info("New Status: " + status)
        
    @skip_if_sim(default_return=True)
    def register_write(self, REGISTER_NUMBER: int, VALUE: int | float) -> bool:
        # Command set is built up by: Start char - command - data - stop char
        # Start Char "$""
        # Command "R41=" - register reference =
        # Data "23.5" for example
        # Stop char <CR> - carriage return '0x0D' or '\r'

        # When the regulator is ready to receive next command
        # we get following ASCII chars: 'CR' 'LF' '>' 'SP' 
        # HEX 0D 0A 3E 20

        # Each character sent to the regulator is echoed back from the regulator
        # and when the end of the command (CR) has been sent
        # the regulator responds with ASCII chars: 'CR' 'LF' followed by response as the table below. 
        # Note that after each command and response, following string is sent in ASCII chars: 
        # 'CR' 'LF' '>' 'SP' (HEX 0D 0A 3E 20).
        # if sent command is unknown, the regulator responds with '?' + command
        # Note commands are case sensitive

        # For RXX=, if data=int response is <Downloaded data>, if foat <no response>
        
        msg = f"$R{REGISTER_NUMBER}={VALUE}"

        self.ser.write((msg+'\r').encode('ascii'))

        repeat = self.get_data().split(" ")[1]
        response = self.get_data()
        if (response == f"{VALUE}" or response == '') and repeat == msg:
            #logging.info(f"Successfully wrote to R{REGISTER_NUMBER}.")
            return True
        else:
            logging.error(f"Failed to write to R{REGISTER_NUMBER}.")
            return False
        
    @skip_if_sim(default_return=0)
    def register_read(self, REGISTER_NUMBER: int) -> float | int:
        msg = f"$R{REGISTER_NUMBER}?"
        self.ser.write((msg+'\r').encode('ascii'))

        repeat = self.get_data().split(" ")[1]
        if repeat != msg:
            logging.error(f"Failed to read R{REGISTER_NUMBER}.")

        return float(self.get_data())
        
    def set_regulator_mode(self, mode: int = 6) -> bool:
        # 1 = Power 
        # 2 = ON/OFF
        # 3 = P 
        # 4 = PI 
        # 5 = PD 
        # 6 = PID 
        
        return self.register_write(13, mode)

    def clamp(self, n: int | float , minn: int | float, maxn: int | float) -> int | float:
        if n < minn:
            return minn
        elif n > maxn:
            return maxn
        else:
            return n

    def set_max_tc(self, max: float) -> bool:
        # 100 is default register value
        return self.register_write(6, self.clamp(max, 0, 100))
        
    def set_tc_dead_band(self) -> bool:
        return self.register_write(7, self.clamp(self.dead_band, 0, 100))

        # Dead band is limiting the signal around zero value. 
        # Good to adjust if we do not like fast switching from one voltage direction to the other
        # This helps to save the life of the peltier modules

    def set_pid_parameters(self, p: float, i: float, d: float) -> bool:
        if (self.register_write(1, p) is True) and (self.register_write(2, i) is True) and (self.register_write(3, d) is True):
            return True
        else:
            return False

    def set_low_pass(self, low_pass_a: float = 2, low_pass_b: float = 3) -> bool:
        # Default controller values for now => no need to set
        if self.register_write(4, abs(low_pass_a)) is True and self.register_write(5, abs(low_pass_b)) is True:
            return True
        else:
            return False

    def set_fan_modes(self, mode: int = 4) -> None:
        # Always OFF = 0
        # Always ON = 1
        # Cool = 2
        # Heat = 3
        # Cool / Heat = 4, on when main output is non zero (reg[106])
        
        if (self.register_write(16, mode) is True) and (self.register_write(23, mode) is True) and (self.register_write(22, self.fan_voltage) is True) and (self.register_write(29, self.fan_voltage) is True):
            logging.info("Temperature regulator fan settings successfully configured.")
        else:
            logging.error("Temperature regulator fan configuration failed.")
            sys.exit()
        
    def turn_fans_off(self) -> None:
        # To be used when taking mass readings
        if (self.register_write(16, 0) is True) and (self.register_write(23, 0) is True):
            logging.info("Fans successfully turned off.")
        else:
            logging.error("Failed to turn off fans.")
        
    def set_voltage_alarm_settings(self) -> bool:
        # Set alarms for over and under voltage
        if (self.register_write(45, self.input_voltage + 1) is True) and (self.register_write(46, self.input_voltage - 1) is True):
            return True
        else:
            return False
        
    def set_current_alarm_settings(self) -> bool:
        # Main current under and over
        # Fan current over
        if (self.register_write(47, self.max_current) is True) and (self.register_write(48, self.min_current) is True) and (self.register_write(49, self.fan_current) is True) and (self.register_write(51, self.fan_current) is True):
            return True
        else:
            return False
        
    def configure_main_sensor(self, mode: int = 12) -> bool:
        # 2 = to activate Steinhart calculation
        # 3 = to activate Zoom mode (internal control of digital pot). Have this bit set to achieve maximal resolution.
        # 4 = to activate PT mode 

        # To revisit best mode and values to use

        # Also set alarms on over and under

        if (self.register_write(55, mode) is True) and (self.register_write(71, self.max_temp + 5) is True) and (self.register_write(72, self.min_temp - 5) is True):
            return True
        else:
            return False
        
    def configure_heat_sink_sensor(self, mode: int = 12) -> bool:
        # 2 = to activate Steinhart calculation
        # 3 = to activate Zoom mode (internal control of digital pot). Have this bit set to achieve maximal resolution.
        # 4 = to activate PT mode 

        # Also set alarms on over and under
        if (self.register_write(56, mode) is True) and (self.register_write(73, self.max_temp + 5) is True) and (self.register_write(74, self.min_temp - 5) is True):
            return True
        else:
            return False
        
    def set_main_steinhart_coeffs(self) -> bool:
        if (self.register_write(59, self.A_coeff_2) is True) and (self.register_write(60, self.B_coeff_2) is True) and (self.register_write(61, self.C_coeff_2) is True):
            return True
        else:
            return False
        
    def set_heat_sink_steinhart_coeffs(self) -> bool:
        if (self.register_write(62, self.A_coeff_2) is True) and (self.register_write(63, self.B_coeff_2) is True) and (self.register_write(64, self.C_coeff_2) is True):
            return True
        else:
            return False
    
    def get_t1_mode(self) -> int:
        return self.register_read(55)
    
    def get_t2_mode(self) -> int:
        return self.register_read(65)
                
    def get_tc_value(self) -> float:
        return self.register_read(106)
    
    def get_t1_value(self) -> float:
        return self.register_read(100)
    
    def get_t2_value(self) -> float:
        return self.register_read(101)
    
    def get_main_current(self) -> float:
        return self.register_read(152)
    
    def get_fan1_current(self) -> float:
        return self.register_read(153)
    
    def get_fan2_current(self) -> float:
        return self.register_read(154)
    
    def set_heating_mode(self) -> None:
        if (self.set_max_tc(self.heating_tc) is True) and (self.set_pid_parameters(self.heating_Kp, self.heating_Ki, self.heating_Kd) is True):
                logging.info("Temperature regulator set to heating mode.")
        else:
            logging.error("Failed to set temperature regulator to heating mode.")
            sys.exit()

    def set_cooling_mode(self) -> None:
        if (self.set_max_tc(self.cooling_tc) is True) and (self.set_pid_parameters(self.cooling_Kp, self.cooling_Ki, self.cooling_Kd) is True):
                logging.info("Temperature regulator set to cooling mode.")
        else:
            logging.error("Failed to set temperature regulator to cooling mode.")
            sys.exit()

    def set_subzero_mode(self) -> None:
        if (self.set_max_tc(self.subzero_tc) is True) and (self.set_pid_parameters(self.subzero_Kp, self.subzero_Ki, self.subzero_Kd) is True):
                logging.info("Temperature regulator set to subzero mode.")
        else:
            logging.error("Failed to set temperature regulator to subzero mode.")
            sys.exit()

    def set_temperature(self, temp: float) -> None:
        self.assess_status()
        
        if temp >= self.temp_threshold:
            self.set_heating_mode()
        elif temp >= self.subzero_threshold:
            self.set_cooling_mode()
        else:
            self.set_subzero_mode()

        if self.register_write(0, self.clamp(temp, self.min_temp, self.max_temp)) is True:
            logging.info(f"Peltier target temperature set to {temp}C.")
        else:
            logging.error("Failed to set peltier target temperature.")
            sys.exit()

        self.set_run_flag()
    
    @skip_if_sim(default_return=True)
    def wait_until_temperature(self, 
                            value: float,
                            show_temperature_fn: Callable[[str], None],
                            sample_rate: float = 30, 
                            keep_on: bool = True,
                            ) -> bool:      
          
        self.set_temperature(value)
        temperature = self.get_t1_value()

        start = time.time()
        elapsed_time = 0
        count = 0

        max_count = floor(self.steady_state / sample_rate) - 1

        while (elapsed_time < self.timeout) and (count < max_count):
            elapsed_time = time.time() - start
            temperature = self.get_t1_value()

            logging.info(f"Temperature progress is {round(temperature, 2)}/{value}C.")
            logging.info(f"TEC Power: {round(self.get_tc_value(), 1)}% PWM -> {round(self.get_main_current(), 1)}A.")

            # Check steady state based on number of counts (depends on sample rate)
            if abs(value - temperature) <= self.allowable_error:
                count += 1
                logging.info(f"Error < allowable error (count = {count})")
            else:
                count = 0

            local_start = time.time()
            while time.time() - local_start < sample_rate:
                show_temperature_fn(f"Cell Temperature: {self.get_t1_value():.1f}C @ {self.get_tc_value():.1f}% Power")
                time.sleep(5)

        if count >= max_count:
            logging.info(f"Temperature controller successfully reached {value}C in {time.time() - start}s.")
        
            # Turn controller OFF if required
            if keep_on is False:
                logging.info("Switching off TEC.")
                self.clear_run_flag()

            return True
        
        logging.error(f"Temperature controller timed out trying to reach {value}C.")
        logging.info(f"Final TEC Power: {round(self.get_tc_value(), 1)}% PWM -> {round(self.get_main_current(), 1)}A.")

        # Turn controller OFF
        self.clear_run_flag()
        return False
    
    @skip_if_sim(default_return=True)
    def plot_live_temperature_control(self, value: float, sample_rate: float = 1) -> bool:        
        self.set_temperature(value)
        global_start = time.time()

        plt.ion()
        plot_width = self.timeout * sample_rate

        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(111)
            
        plt.title(f"Target Temp: {value}C, Sample Rate: {sample_rate}Hz")
        plt.suptitle("Live Data:")
        plt.xlabel("Samples")
        plt.ylim([-100, 100])
        plt.grid(visible=True, which="both", axis="both")

        error = [0] * plot_width
        dT = [0] * plot_width
        drive = [0] * plot_width
        samples = range(1, plot_width+1)

        line1, = ax.plot(samples, error, 'r-', label="Temperature Error K")
        line2, = ax.plot(samples, drive, 'g-', label="Drive Power %")
        line3, = ax.plot(samples, dT, 'b-', label="dT K")
        plt.legend(loc="upper right")

        # Turn controller ON
        self.set_run_flag()

        while (time.time() - global_start) < self.timeout:

            temperature = self.get_t1_value()
            sink = self.get_t2_value()
            curr = self.get_main_current()
            fan_curr = self.get_fan1_current() + self.get_fan2_current()         

            # Append and loose first element
            plt.title(f"Target Temp: {value}C, Sample Rate: {sample_rate}Hz")
            plt.suptitle(f"Live Data: Control Temperature = {round(temperature,2)}C, Heat Sink Temperature = {round(sink,2)}C, Main Current = {round(curr,2)}A, Fan Current = {round(fan_curr,2)}A, Elapsed Time = {round(time.time() - global_start,2)}s")

            error.append(value - temperature)
            error = error[-plot_width:]

            drive.append(self.get_tc_value())
            drive = drive[-plot_width:]

            dT.append(abs(temperature - sink))
            dT = dT[-plot_width:]

            line1.set_ydata(error)
            line2.set_ydata(drive)
            line3.set_ydata(dT)

            fig.canvas.draw()
            fig.canvas.flush_events()
                
            local_start = time.time()     

            while (abs(value - self.get_t1_value()) < self.allowable_error) and (time.time() - local_start < self.steady_state):
                time.sleep(1 / sample_rate)

            # Check if steady state timeout reached
            if (time.time() - local_start) >= self.steady_state:
                logging.info(f"Temperature controller successfully reached {value}C in {time.time() - global_start}s")
                logging.info(f"Final peltier current is {round(self.get_main_current(),2)}A.")

                # Turn controller OFF
                self.clear_run_flag()
                return True
            
            time.sleep(1 / sample_rate)
            
        logging.error(f"Temperature controller timed out trying to reach {value}C.")
        logging.info(f"Final peltier current is {round(self.get_main_current(),2)}A.")

        # Turn controller OFF
        self.clear_run_flag()
        return False