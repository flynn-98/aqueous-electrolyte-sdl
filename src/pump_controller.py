import logging
import sys

import serial

logging.basicConfig(level = logging.INFO)

def skip_if_sim(default_return = None):
    def decorator(func):
        def wrapper(self: pump_controller, *args, **kwargs):
            if self.sim:
                return default_return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

class pump_controller:
    def __init__(self, COM: str, baud: int = 115200, sim: bool = False):
        self.sim = sim

        if self.sim:
            logging.info("Simulated connection to pump controller established.")

        else:
            logging.info("Configuring pump controller serial port..")
            self.ser = serial.Serial(COM) 
            self.ser.baudrate = baud
            self.ser.bytesize = 8 
            self.ser.parity = 'N' # No parity
            self.ser.stopbits = 1

            logging.info("Attempting to open pump controller serial port..")

            if self.ser.isOpen() is False:
                self.ser.open()

            # Check connection (blocking)
            if self.check_status():
                logging.info("Serial connection to pump controller established.")

    def get_data(self) -> str:
        while self.ser.in_waiting == 0:
            pass

        return self.ser.readline().decode().rstrip().replace("\x00", "")
        
    def check_response(self) -> None:
        while(1):
            data = self.get_data()
            # Wait for response and check that command was understood
            if '#' in data:
                break
            elif "Unknown command" in data:
                logging.error("Pump controller failed to recognise command: " + data)
                sys.exit()
            else:
                logging.info("Response from pump controller: " + data)

    @skip_if_sim()
    def close_ser(self) -> None:
        logging.info("Closing serial connection to pump controller.")
        if self.ser.isOpen():
            self.ser.close()

    @skip_if_sim(default_return = True)
    def check_status(self) -> bool:
        self.ser.write("statusCheck()".encode())
        self.check_response()
        
    @skip_if_sim()
    def display_oled_message(self, msg: str) -> None:
        self.ser.write(f"displayMessage({msg})".encode())
        self.check_response()

    @skip_if_sim(default_return = 25)
    def get_temperature(self) -> float:
        self.ser.write("getTemperature()".encode())
        return float(self.get_data())
        
    @skip_if_sim(default_return = 50)
    def get_humidity(self) -> float:
        self.ser.write("getHumidity()".encode())
        return float(self.get_data())
        
    @skip_if_sim()
    def single_pump(self, pump_no: int, volume: float) -> None:
        self.ser.write(f"singleStepperPump({pump_no},{volume:.4f})".encode())
        self.check_response()

    @skip_if_sim()
    def multi_pump(self, volumes: list[float], check: bool = True) -> None:
        if len(volumes) != 4:
            raise ValueError("Exactly 4 volumes are required")
        
        args = ",".join(f"{float(v):.4f}" for v in volumes)
        self.ser.write(f"multiStepperPump({args})".encode())

        if check:
            self.check_response()

    @skip_if_sim()
    def transfer_pump(self, pump_no: int, pwm: float, seconds: float, check: bool = True) -> None:
        self.ser.write(f"transferPump({pump_no},{pwm},{seconds})".encode())

        if check:
            self.check_response()