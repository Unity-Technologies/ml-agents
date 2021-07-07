import os
import psutil
import pickle
import time

from multiprocessing import Process, Pipe
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class UtilsTracker:
    def __init__(self, worker_pids, save_path):
        self.proc = None
        self.conn = None
        self.pids = [os.getpid()] + worker_pids
        self.output_path = os.path.join(save_path, "utils_summaries.pkl")
        self._create_track_process(self.pids)

    def _create_track_process(self, pids):
        parent_conn, child_conn = Pipe()
        self.conn = parent_conn
        self.proc = Process(target=_track, args=(pids, child_conn, self.output_path))
        self.proc.start()

    def shutdown(self):
        try:
            self.conn.send("TERMINATE")
            self.proc.join()
        except Exception:
            pass


def _track(pids, parent_conn, output_path):
    buffers = {"sys": [], "process": {str(pid): [] for pid in pids}}
    util_procs = [psutil.Process(pid) for pid in pids]
    process_names = ["main"] + list(range(len(util_procs) - 1))
    try:
        buffers["meta"] = {
            "cpu_count_logical": psutil.cpu_count(),
            "cpu_count": psutil.cpu_count(logical=True),
        }
        while True:
            if parent_conn.poll(0) is True and parent_conn.recv() == "TERMINATE":
                break
            # system-wise stats
            try:
                data = {
                    "cpu_percent": psutil.cpu_percent(),
                    "cpu_percent_percpu": psutil.cpu_percent(percpu=True),
                    "memory": psutil.virtual_memory(),
                    "timestamp": time.time(),
                }
                buffers["sys"].append(data)
            except Exception as e:
                print(e)
            # per-process stats
            for p, name in zip(util_procs, process_names):
                try:
                    data = p.as_dict(
                        attrs=["cpu_percent", "cpu_times", "memory_percent"]
                    )
                    data["timestamp"] = time.time()

                    buffers["process"][name].append(data)
                except Exception as e:
                    print(e)
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.debug("util tracker interrupted")
    except Exception as e:
        logger.debug(f"util tracker error: {e}")
    finally:
        with open(output_path, "wb") as f:
            pickle.dump(buffers, f)
        parent_conn.close()
