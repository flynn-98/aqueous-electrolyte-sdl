# pump_controller_ble.py
import asyncio
import logging
import sys
from collections import deque
from typing import Optional, List
from bleak import BleakScanner, BleakClient
import threading
import concurrent.futures

logging.basicConfig(level=logging.INFO)

def skip_if_sim(default_return=None):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if getattr(self, "sim", False):
                return default_return
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

DEVICE_NAME_DEFAULT = "PumpControllerA"

class PumpControllerBLE:
    """
    BLE-backed controller with the same external API shape as the serial PumpController.
    - Methods accept `check: bool` to defer response handling (for multi-device sync).
    - `check_response()` pulls queued notifications until a '#' line appears.
    - `sim` flag short-circuits methods via @skip_if_sim, mirroring the serial driver.
    """
    def __init__(self, device_name: str = DEVICE_NAME_DEFAULT, sim: bool = False, timeout: float = 60.0):
        self.device_name = device_name
        self.sim = sim
        self.timeout = timeout

        # BLE state
        self.client: Optional[BleakClient] = None
        self._char_uuid: Optional[str] = None
        self._loop = asyncio.get_event_loop()
        self._inbox: deque[str] = deque()
        self._notify_event = asyncio.Event()

        # Private event loop running in a background thread (works in Jupyter)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        if not self.sim:
            self._call(self._connect())  # run _connect() on private loop (for juypter notebook compatibility)
            self.status_check()

    def _call(self, coro):
        """Run an async coroutine on the private loop and return its result."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result()
        except concurrent.futures.TimeoutError:
            return None
    
    async def _connect(self):
        print(f"Scanning for BLE device: {self.device_name}")
        dev = await BleakScanner.find_device_by_name(self.device_name, timeout=10.0)
        if not dev:
            raise RuntimeError(f"Device '{self.device_name}' not found / not advertising")

        self.client = BleakClient(dev)
        await self.client.connect()
        print("BLE Connected")

        # ---- Robust service discovery across Bleak versions ----
        svcs = getattr(self.client, "services", None)

        # try new API first
        if (not svcs) or (len(list(svcs)) == 0):
            if hasattr(self.client, "get_services_cache"):
                await self.client.get_services_cache()
                svcs = self.client.services

        # then old API
        if (not svcs) or (len(list(svcs)) == 0):
            if hasattr(self.client, "get_services"):
                # returns a collection on older Bleak; also populates client.services
                svcs = await self.client.get_services()

        # final nudge
        if (not svcs) or (len(list(svcs)) == 0):
            await asyncio.sleep(0.3)
            svcs = self.client.services

        if (not svcs) or (len(list(svcs)) == 0):
            raise RuntimeError("Failed to resolve GATT services after connect")

        # ---- Pick first char with notify AND (write or write-without-response) ----
        cand_uuid = None
        for s in svcs:
            for c in s.characteristics:
                props = set(c.properties or [])
                if "notify" in props and ("write" in props or "write-without-response" in props):
                    cand_uuid = str(c.uuid)
                    break
            if cand_uuid:
                break

        if not cand_uuid:
            self._dump_gatt(svcs)
            raise RuntimeError("No characteristic with notify+write found on device")

        self._char_uuid = cand_uuid

        # Subscribe first; small delay helps CoreBluetooth/macOS
        await self.client.start_notify(self._char_uuid, self._on_notify)
        await asyncio.sleep(0.15)


    def _on_notify(self, _handle: int, data: bytearray):
        msg = data.decode("utf-8", errors="ignore").strip()
        if msg:
            self._inbox.append(msg)
        if not self._notify_event.is_set():
            self._notify_event.set()

    # ----------------- Core I/O -----------------

    async def _write_async(self, payload: str):
        if not self.client or not self.client.is_connected or not self._char_uuid:
            raise RuntimeError("BLE not connected")
        # Prefer write-with-response; Bleak will fall back if unsupported
        await self.client.write_gatt_char(self._char_uuid, payload.encode("utf-8"), response=True)

    async def _read_async(self, timeout: Optional[float] = None) -> Optional[str]:
        if self._inbox:
            return self._inbox.popleft()
        self._notify_event.clear()
        try:
            await asyncio.wait_for(self._notify_event.wait(), timeout or self.timeout)
        except asyncio.TimeoutError:
            return None
        return self._inbox.popleft() if self._inbox else None

    def get_data(self, timeout: Optional[float] = None) -> Optional[str]:
        """Sync wrapper: return next notification string, or None on timeout."""
        return self._call(self._read_async(timeout))

    def check_response(self) -> None:
        """Block until a line containing '#' arrives; log others like the serial driver."""
        while True:
            data = self.get_data(timeout=self.timeout)
            if data is None:
                logging.warning("BLE: timed out waiting for response")
                break
            if '#' in data:
                break
            elif "Unknown command" in data:
                raise RuntimeError("Pump controller failed to recognise command: " + data)
            else:
                logging.info("Response from pump controller: " + data)

    # Compatibility alias (some schedulers prefer plural)
    def check_responses(self) -> None:
        """Compatibility alias used by some schedulers."""
        return self.check_response()

    # ----------------- Public API (mirror serial) -----------------

    @skip_if_sim()
    def status_check(self) -> None:
        self._call(self._write_async("statusCheck()"))
        self.check_response()

    @skip_if_sim(default_return=25)
    def get_temperature(self) -> Optional[float]:
        self._call(self._write_async("getTemperature()"))
        data = self.get_data(timeout=self.timeout)
        try:
            return float(data) if data is not None else None
        except (TypeError, ValueError):
            return None

    @skip_if_sim(default_return=50)
    def get_humidity(self) -> Optional[float]:
        self._call(self._write_async("getHumidity()"))
        data = self.get_data(timeout=self.timeout)
        try:
            return float(data) if data is not None else None
        except (TypeError, ValueError):
            return None

    @skip_if_sim()
    def display_oled_message(self, message: str) -> None:
        self._call(self._write_async(f"displayMessage({message})"))
        self.check_response()

    @skip_if_sim()
    def single_pump(self, pump_no: int, ml: float, flow_rate: float = 0.05, check: bool = True) -> None:
        self._call(self._write_async(f"singleStepperPump({pump_no},{ml:.3f},{flow_rate:.3f})"))
        if check:
            self.check_response()

    @skip_if_sim()
    def multi_pump(self, ml: List[float], flow_rate: float = 0.05, check: bool = True) -> None:
        if len(ml) != 4:
            raise ValueError("Exactly 4 volumes are required")
        
        args = ",".join(f"{float(v):.3f}" for v in ml)
        self._call(self._write_async(f"multiStepperPump({args},{flow_rate:.3f})"))
        if check:
            self.check_response()

    @skip_if_sim()
    def transfer_pump(self, pump_no: int, pwm: float, seconds: float, check: bool = True) -> None:
        self._call(self._write_async(f"transferPump({pump_no},{pwm},{seconds})"))
        if check:
            self.check_response()

    # Serial-compat close name
    @skip_if_sim()
    def close_ser(self) -> None:
        self.close()

    def close(self) -> None:
        async def _close():
            if self.client and self.client.is_connected:
                try:
                    if self._char_uuid:
                        try: await self.client.stop_notify(self._char_uuid)
                        except Exception: pass
                finally:
                    try: await self.client.disconnect()
                    except Exception: pass
            await asyncio.sleep(0.4)  # let CoreBluetooth release

        if not self.sim:
            self._call(_close())

        # stop the private event loop
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_loop_thread"):
            self._loop_thread.join(timeout=1.0)

    # ---------- Debug helper ----------
    def _dump_gatt(services):
        print("---- GATT on device ----")
        for s in services:
            print(f"Service {s.uuid}")
            for c in s.characteristics:
                try:
                    props = ",".join(c.properties)
                except Exception:
                    props = ""
                print(f"  Char {c.uuid} [{props}]")
        print("------------------------")


if __name__ == "__main__":
    ctrl = PumpControllerBLE()
    print("Status:", ctrl.status_check())
    #print("Temp:", ctrl.get_temperature())
    #print("Hum:", ctrl.get_humidity())
    #ctrl.display_oled_message("Hello BLE")
    #ctrl.transfer_pump(1, 70, 5)
    ctrl.single_pump(1, 1.0)
    #ctrl.multi_pump([1.0,2.0,3.0,4.0], 0.05)