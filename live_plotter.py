from src.scheduler import scheduler

recipe_path = "config/electrolyte_recipe.yaml"
config_path = "config/hardware_config.yaml"

device = scheduler(config_path=config_path, recipe_path=recipe_path)
device.tec.plot_live_temperature_control(40)