import sys

plugin_paths = ["/Users/chris.elion/code/mlagents-plugins/mlagentsplugins/plugins"]
for p in plugin_paths:
    sys.path.insert(1, p)

import importlib
import pkgutil

from mlagents.trainers.stats import StatsWriter

original_StatsWriters = set(StatsWriter.__subclasses__())

discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg in pkgutil.iter_modules(plugin_paths)
}

print(discovered_plugins)
all_StatsWriters = set(StatsWriter.__subclasses__())  # finds the new subclass
new_StatsWriters = all_StatsWriters - original_StatsWriters
print(f"Found new StatsWriters: {new_StatsWriters}")

# Instantiate the discovered classes and call them
for cls in new_StatsWriters:
    sw: StatsWriter = cls()
    sw.write_text("category", "I'm a new plugin", 42)
