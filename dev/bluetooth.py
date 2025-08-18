# ble_roundtrip.py
import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "PumpController"
CHAR_UUID   = "42bb8b48-f737-4435-adba-f578eba53675"  # from firmware

async def main():
    dev = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if not dev:
        print("Device not found / not advertising")
        return

    got = asyncio.Event()
    last_msg = {"data": None}

    def on_notify(_, data: bytearray):
        last_msg["data"] = data.decode("utf-8", errors="ignore").strip()
        got.set()

    async with BleakClient(dev) as client:
        await client.start_notify(CHAR_UUID, on_notify)
        await asyncio.sleep(0.15)  # let CCCD settle on macOS
        await client.write_gatt_char(CHAR_UUID, b"statusCheck()", response=True)
        try:
            await asyncio.wait_for(got.wait(), timeout=5.0)
        finally:
            await client.stop_notify(CHAR_UUID)
    # exiting the 'with' guarantees disconnect; add a small pause after
    await asyncio.sleep(0.4)

if __name__ == "__main__":
    asyncio.run(main())
