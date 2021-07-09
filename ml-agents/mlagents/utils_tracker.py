import os
import psutil
import time

from tinydb import TinyDB
from multiprocessing import Process, Pipe
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class UtilsTracker:
    def __init__(self, worker_pids, save_path):
        self.proc = None
        self.conn = None
        self.pids = [os.getpid()] + worker_pids
        self.output_dir = save_path
        self._create_track_process(self.pids)

    def _create_track_process(self, pids):
        parent_conn, child_conn = Pipe()
        self.conn = parent_conn
        self.proc = Process(target=_track, args=(pids, child_conn, self.output_dir))
        self.proc.start()

    def shutdown(self):
        try:
            self.conn.send("TERMINATE")
            self.proc.join()
        except Exception:
            pass


def _track(pids, parent_conn, output_dir):
    util_procs = [psutil.Process(pid) for pid in pids]
    process_names = ["main"] + [f"worker_{i}" for i in range(len(util_procs) - 1)]
    db_meta = TinyDB(os.path.join(output_dir, "sys_utils_meta.json"))
    db = TinyDB(os.path.join(output_dir, "sys_utils.json"))
    try:
        db_meta.insert(
            {
                "cpu_count_logical": psutil.cpu_count(),
                "cpu_count": psutil.cpu_count(logical=True),
                "memory": psutil.virtual_memory().total,
            }
        )
        while True:
            if parent_conn.poll(0) is True and parent_conn.recv() == "TERMINATE":
                break
            stats = {}
            # system-wise stats
            try:
                stats["sys"] = get_sys_stats()
            except Exception as e:
                print("system stats error: ", e)
            # per-process stats
            stats["process"] = {}
            try:
                for p, name in zip(util_procs, process_names):
                    stats["process"][name] = get_process_stats(p)
            except Exception as e:
                print("process stats error: ", e)
            db.insert(stats)
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.debug("util tracker interrupted")
    except Exception as e:
        logger.debug(f"util tracker error: {e}")
    finally:
        parent_conn.close()
        logger.debug("util tracking thread shut down")


def get_sys_stats():
    stats = {}
    stats["cpu_percent"] = psutil.cpu_percent()
    stats["cpu_percent_percpu"] = psutil.cpu_percent(percpu=True)
    mem = psutil.virtual_memory()
    stats["memory"] = {
        "available": mem.available,
        "percent": mem.percent,
        "free": mem.free,
        "active": mem.active,
        "inactive": mem.inactive,
        # "buffers": mem.buffers,
        # "cached": mem.cached,
        # "shared": mem.shared,
        # "slab": mem.slab,
    }
    net = psutil.net_io_counters()
    stats["network_io"] = {"bytes_sent": net.bytes_sent, "bytes_recv": net.bytes_recv}
    stats["timestamp"] = time.time()
    return stats


def get_process_stats(p):
    data = p.as_dict(
        attrs=[
            "cpu_percent",
            "cpu_times",
            "cpu_num",
            "num_threads",
            "memory_info",
            "memory_percent",
        ]
    )
    stats = {}
    stats["cpu_percent"] = data["cpu_percent"]
    stats["cpu_num"] = data["cpu_num"]
    cpu = data["cpu_times"]
    stats["cpu_times"] = {
        "user": cpu.user,
        "system": cpu.system,
        "children_user": cpu.children_user,
        "children_system": cpu.children_system,
        "iowait": cpu.iowait,
    }
    stats["num_threads"] = data["num_threads"]
    stats["memory_percent"] = data["memory_percent"]
    mem = data["memory_info"]
    stats["memory_info"] = {
        "rss": mem.rss,
        "vms": mem.vms,
        "shared": mem.shared,
        "text": mem.text,
        "lib": mem.lib,
        "data": mem.data,
        "dirty": mem.dirty,
    }
    stats["timestamp"] = time.time()
    return stats
