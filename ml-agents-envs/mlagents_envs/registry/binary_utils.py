import urllib.request
import tempfile
import os
import uuid
import shutil
import glob

import yaml
import hashlib

from zipfile import ZipFile
from sys import platform
from typing import Tuple, Optional, Dict, Any

from filelock import FileLock

from mlagents_envs.env_utils import validate_environment_path

from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)

# The default logical block size is 8192 bytes (8 KB) for UFS file systems.
BLOCK_SIZE = 8192


def get_local_binary_path(name: str, url: str, tmp_dir: Optional[str] = None) -> str:
    """
    Returns the path to the executable previously downloaded with the name argument. If
    None is found, the executable at the url argument will be downloaded and stored
    under name for future uses.
    :param name: The name that will be given to the folder containing the extracted data
    :param url: The URL of the zip file
    :param: tmp_dir: Optional override for the temporary directory to save binaries and zips in.
    """
    NUMBER_ATTEMPTS = 5
    tmp_dir = tmp_dir or tempfile.gettempdir()
    lock = FileLock(os.path.join(tmp_dir, name + ".lock"))
    with lock:
        path = get_local_binary_path_if_exists(name, url, tmp_dir=tmp_dir)
        if path is None:
            logger.debug(
                f"Local environment {name} not found, downloading environment from {url}"
            )
        for attempt in range(
            NUMBER_ATTEMPTS
        ):  # Perform 5 attempts at downloading the file
            if path is not None:
                break
            try:
                download_and_extract_zip(url, name, tmp_dir=tmp_dir)
            except Exception:
                if attempt + 1 < NUMBER_ATTEMPTS:
                    logger.warning(
                        f"Attempt {attempt + 1} / {NUMBER_ATTEMPTS}"
                        ": Failed to download and extract binary."
                    )
                else:
                    raise
            path = get_local_binary_path_if_exists(name, url, tmp_dir=tmp_dir)

    if path is None:
        raise FileNotFoundError(
            f"Binary not found, make sure {url} is a valid url to "
            "a zip folder containing a valid Unity executable"
        )
    return path


def get_local_binary_path_if_exists(name: str, url: str, tmp_dir: str) -> Optional[str]:
    """
    Recursively searches for a Unity executable in the extracted files folders. This is
    platform dependent : It will only return a Unity executable compatible with the
    computer's OS. If no executable is found, None will be returned.
    :param name: The name/identifier of the executable
    :param url: The url the executable was downloaded from (for verification)
    :param: tmp_dir: Optional override for the temporary directory to save binaries and zips in.
    """
    _, bin_dir = get_tmp_dirs(tmp_dir)
    extension = None

    if platform == "linux" or platform == "linux2":
        extension = "*.x86_64"
    if platform == "darwin":
        extension = "*.app"
    if platform == "win32":
        extension = "*.exe"
    if extension is None:
        raise NotImplementedError("No extensions found for this platform.")
    url_hash = "-" + hashlib.md5(url.encode()).hexdigest()
    path = os.path.join(bin_dir, name + url_hash, "**", extension)
    candidates = glob.glob(path, recursive=True)
    if len(candidates) == 0:
        return None
    else:
        for c in candidates:
            # Unity sometimes produces another .exe file that we must filter out
            if "UnityCrashHandler64" not in c:
                # If the file is not valid, return None and delete faulty directory
                if validate_environment_path(c) is None:
                    shutil.rmtree(c)
                    return None
                return c
        return None


def _get_tmp_dir_helper(tmp_dir: Optional[str] = None) -> Tuple[str, str]:
    tmp_dir = tmp_dir or ("/tmp" if platform == "darwin" else tempfile.gettempdir())
    MLAGENTS = "ml-agents-binaries"
    TMP_FOLDER_NAME = "tmp"
    BINARY_FOLDER_NAME = "binaries"
    mla_directory = os.path.join(tmp_dir, MLAGENTS)
    if not os.path.exists(mla_directory):
        os.makedirs(mla_directory)
        os.chmod(mla_directory, 16877)
    zip_directory = os.path.join(tmp_dir, MLAGENTS, TMP_FOLDER_NAME)
    if not os.path.exists(zip_directory):
        os.makedirs(zip_directory)
        os.chmod(zip_directory, 16877)
    bin_directory = os.path.join(tmp_dir, MLAGENTS, BINARY_FOLDER_NAME)
    if not os.path.exists(bin_directory):
        os.makedirs(bin_directory)
        os.chmod(bin_directory, 16877)
    return zip_directory, bin_directory


def get_tmp_dirs(tmp_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Returns the path to the folder containing the downloaded zip files and the extracted
    binaries. If these folders do not exist, they will be created.
    :retrun: Tuple containing path to : (zip folder, extracted files folder)
    """
    # TODO: Once we don't use python 3.7 we should just use exists_ok=True when creating the dirs to avoid this.
    # Should only be able to error out 3 times (once for each subdir).
    for _attempt in range(3):
        try:
            return _get_tmp_dir_helper(tmp_dir)
        except FileExistsError:
            continue
    return _get_tmp_dir_helper(tmp_dir)


def download_and_extract_zip(
    url: str, name: str, tmp_dir: Optional[str] = None
) -> None:
    """
    Downloads a zip file under a URL, extracts its contents into a folder with the name
    argument and gives chmod 755 to all the files it contains. Files are downloaded and
    extracted into special folders in the temp folder of the machine.
    :param url: The URL of the zip file
    :param name: The name that will be given to the folder containing the extracted data
    :param: tmp_dir: Optional override for the temporary directory to save binaries and zips in.
    """
    zip_dir, bin_dir = get_tmp_dirs(tmp_dir)
    url_hash = "-" + hashlib.md5(url.encode()).hexdigest()
    binary_path = os.path.join(bin_dir, name + url_hash)
    if os.path.exists(binary_path):
        shutil.rmtree(binary_path)

    # Download zip
    try:
        request = urllib.request.urlopen(url, timeout=30)
    except urllib.error.HTTPError as e:  # type: ignore
        e.reason = f"{e.reason} {url}"  # type: ignore
        raise
    zip_size = int(request.headers["content-length"])
    zip_file_path = os.path.join(zip_dir, str(uuid.uuid4()) + ".zip")
    with open(zip_file_path, "wb") as zip_file:
        downloaded = 0
        while True:
            buffer = request.read(BLOCK_SIZE)
            if not buffer:
                # There is nothing more to read
                break
            downloaded += len(buffer)
            zip_file.write(buffer)
            downloaded_percent = downloaded / zip_size * 100
            print_progress(f"  Downloading {name}", downloaded_percent)
        print("")

    # Extraction
    with ZipFileWithProgress(zip_file_path, "r") as zip_ref:
        zip_ref.extract_zip(f"  Extracting  {name}", binary_path)  # type: ignore
    print("")

    # Clean up zip
    print_progress(f"  Cleaning up {name}", 0)
    os.remove(zip_file_path)

    # Give permission
    for f in glob.glob(binary_path + "/**/*", recursive=True):
        # 16877 is octal 40755, which denotes a directory with permissions 755
        os.chmod(f, 16877)
    print_progress(f"  Cleaning up {name}", 100)
    print("")


def print_progress(prefix: str, percent: float) -> None:
    """
    Displays a single progress bar in the terminal with value percent.
    :param prefix: The string that will precede the progress bar.
    :param percent: The percent progression of the bar (min is 0, max is 100)
    """
    BAR_LEN = 20
    percent = min(100, max(0, percent))
    bar_progress = min(int(percent / 100 * BAR_LEN), BAR_LEN)
    bar = "|" + "\u2588" * bar_progress + " " * (BAR_LEN - bar_progress) + "|"
    str_percent = "%3.0f%%" % percent
    print(f"{prefix} : {bar} {str_percent} \r", end="", flush=True)


def load_remote_manifest(url: str) -> Dict[str, Any]:
    """
    Converts a remote yaml file into a Python dictionary
    """
    tmp_dir, _ = get_tmp_dirs()
    try:
        request = urllib.request.urlopen(url, timeout=30)
    except urllib.error.HTTPError as e:  # type: ignore
        e.reason = f"{e.reason} {url}"  # type: ignore
        raise
    manifest_path = os.path.join(tmp_dir, str(uuid.uuid4()) + ".yaml")
    with open(manifest_path, "wb") as manifest:
        while True:
            buffer = request.read(BLOCK_SIZE)
            if not buffer:
                # There is nothing more to read
                break
            manifest.write(buffer)
    try:
        result = load_local_manifest(manifest_path)
    finally:
        os.remove(manifest_path)
    return result


def load_local_manifest(path: str) -> Dict[str, Any]:
    """
    Converts a local yaml file into a Python dictionary
    """
    with open(path) as data_file:
        return yaml.safe_load(data_file)


class ZipFileWithProgress(ZipFile):
    """
    This is a helper class inheriting from ZipFile that allows to display a progress
    bar while the files are being extracted.
    """

    def extract_zip(self, prefix: str, path: str) -> None:
        members = self.namelist()
        path = os.fspath(path)
        total = len(members)
        n = 0
        for zipinfo in members:
            self.extract(zipinfo, path, None)  # type: ignore
            n += 1
            print_progress(prefix, n / total * 100)
