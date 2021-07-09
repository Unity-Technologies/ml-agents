import os
from typing import Optional
from tinydb import TinyDB

from mlagents.trainers.stats import TensorboardWriter


def write_sys_stats_to_tb(
    utils_output_dir: str, tb_writer: Optional[TensorboardWriter]
) -> None:
    assert tb_writer is not None
    utils_db = TinyDB(os.path.join(utils_output_dir, "sys_utils.json"))
    for data in utils_db:
        for sw in tb_writer.summary_writers.values():
            # system stats
            sw.add_scalar(
                "CPU/System (%)",
                data["sys"]["cpu_percent"],
                walltime=data["sys"]["timestamp"],
            )
            sw.add_scalar(
                "Memory/System (%)",
                data["sys"]["memory"]["percent"],
                walltime=data["sys"]["timestamp"],
            )
            sw.add_scalar(
                "Network/Received (bytes)",
                data["sys"]["network_io"]["bytes_recv"],
                walltime=data["sys"]["timestamp"],
            )
            sw.add_scalar(
                "Network/Sent (bytes)",
                data["sys"]["network_io"]["bytes_sent"],
                walltime=data["sys"]["timestamp"],
            )

            # process stats
            for name, stats in data["process"].items():
                sw.add_scalar(
                    f"CPU/{name} (%)", stats["cpu_percent"], walltime=stats["timestamp"]
                )
                sw.add_scalar(
                    f"CPU/{name} num threads",
                    stats["num_threads"],
                    walltime=stats["timestamp"],
                )
                sw.add_scalar(
                    f"Memory/{name} (%)",
                    stats["memory_percent"],
                    walltime=stats["timestamp"],
                )
