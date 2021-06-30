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
        self._create_track_process(self.pids, save_path)

    def _create_track_process(self, pids, save_path):
        parent_conn, child_conn = Pipe()
        self.conn = parent_conn
        self.proc = Process(target=_track, args=(pids, child_conn, save_path))
        self.proc.start()

    def shutdown(self):
        try:
            self.conn.send("TERMINATE")
            self.proc.join()
        except Exception:
            pass


def _track(pids, parent_conn, save_path):
    buffers = {pid: [] for pid in pids}
    util_procs = [psutil.Process(pid) for pid in pids]
    try:
        while True:
            if parent_conn.poll(0) is True and parent_conn.recv() == "TERMINATE":
                break
            for pid, p in zip(pids, util_procs):
                try:
                    data = p.as_dict(
                        attrs=["cpu_percent", "cpu_times", "memory_percent"]
                    )
                    data["timestamp"] = time.time()
                    buffers[pid].append(data)
                except Exception as e:
                    print(e)
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.debug("util tracker interrupted")
    except Exception as e:
        logger.debug(f"util tracker error: {e}")
    finally:
        with open(os.path.join(save_path, "utils_summaries.pkl"), "wb") as f:
            pickle.dump(buffers, f)
        parent_conn.close()
