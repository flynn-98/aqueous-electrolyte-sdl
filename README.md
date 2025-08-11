# Aqueous Electrolyte SDL
Python repository for controlling and collecting data from the 'PumpBot' self-driving lab - designed to handle aqueous, high concentration electrolytes. 

## Introduction

..

### Be sure to follow all the steps to set up the virtual environment!

## Instal Dependencies

Build venv in root directory:

```
python -m venv .venv
```

Upgrade pip:

```
.venv/bin/pip install --upgrade pip
```

Install dependencies into new venv:

```
.venv/bin/pip install -e .
```

Activate venv:

```
source .venv/bin/activate
```

Note: Replace *bin* with *Scripts* if using windows.

## Set up SquidstatPyLibrary

Download latest *.whl* file from [here](https://github.com/Admiral-Instruments/AdmiralSquidstatAPI/tree/main/SquidstatLibrary/mac/pythonWrapper). Move the file to the repository root directory and run the following command:

```
.venv/bin/pip install FILE.whl
```

You can find documentation for the SquidstatPyLibrary [here](https://admiral-instruments.github.io/AdmiralSquidstatAPI/). Note the Mac ARM64 (Apple Silicon) users must complete [further steps](https://admiral-instruments.github.io/AdmiralSquidstatAPI/md__markdown_files_2_setup_python.html#autotoc_md35:~:text=Mac%20ARM64%20(Apple%20Silicon)) to install.

## Download ATEN RS232 to USB Driver

The Laird PID temperature controller uses RS232 to communicate, which requires an [adapter](https://www.aten.com/global/en/products/usb-solutions/converters/uc232a1/) to convert the signal to a 5V logic level for USB. Download the correct driver [here](https://www.aten.com/global/en/supportcenter/info/downloads/?action=display_product&pid=1142). Installing the drivers may require you to restart your computer.

## Set up Atinary SDK

Download latest *.tar.gz* file from [here](https://enterprise.atinary.com/download/). Move the file to */electrolyte-mixing-station* and run the following command:

```
.venv/bin/pip install FILE.tar.gz
```

You can find documentation for the Atinary SDK [here](https://enterprise.atinary.com/documentation/docs/sdlabs_sdk/installation.html).

## Install Atinary SDL Wrapper

The SDL Wrapper is a codebase created by Atinary, for quick and easy set up of optimisation campaigns using a json config file. Install via the following command:

```
.venv/bin/pip install git+https://github.com/Atinary-technologies/sdlabs_wrapper.git
```

## Recommended Extensions

For easy viewing and editing of CSVs, it is recommended that you download [this CSV extension](https://marketplace.visualstudio.com/items?itemName=ReprEng.csv) for VS Code.

## References
1. [Laird Temperature Controller](https://lairdthermal.com/products/product-temperature-controllers/tc-xx-pr-59-temperature-controller?creative=&keyword=&matchtype=&network=x&device=c&gad_source=1&gclid=CjwKCAiAzPy8BhBoEiwAbnM9O_ueQ3Ph8NvZ4LYCpqO9oUzX78J1sfagfGnYWUDeDpQ8P9rKzc11pBoCUR8QAvD_BwE)
2. [PCX Peltier Module](https://lairdthermal.com/products/thermoelectric-cooler-modules/peltier-thermal-cycling-pcx-series)
3. [Boxer Pump](https://www.boxerpumps.com/peristaltic-pumps-for-liquid/29qq/)
4. [Atinary Self-Driving Labs](https://scientia.atinary.com/sdlabs/academic/dashboard)
5. [Squidstat API Manual](https://admiral-instruments.github.io/AdmiralSquidstatAPI/index.html)