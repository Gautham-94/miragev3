"""Downloads and caches the go2rtc binary for the host platform.

go2rtc (https://github.com/AlexxIT/go2rtc) is a separate Go binary; mirage does not vendor
it. Frigate bundles it at container-build time -- since mirage runs directly on the host
(no container build step), this module instead downloads the correct release asset on
first use and caches it under GO2RTC_BIN_DIR, mirroring how the ONNX model is fetched once
and cached locally rather than vendored into the repo.

Release asset naming confirmed directly against the go2rtc GitHub releases API (not
guessed): Linux assets are raw executables (no extension); macOS/Windows/FreeBSD assets
are zip archives internally containing a single `go2rtc` (or `go2rtc.exe`) binary.
"""

from __future__ import annotations

import io
import logging
import platform
import stat
import urllib.request
import zipfile
from pathlib import Path

from mirage.const import GO2RTC_BIN_DIR

logger = logging.getLogger(__name__)

GO2RTC_VERSION = "v1.9.14"
GO2RTC_RELEASE_BASE = f"https://github.com/AlexxIT/go2rtc/releases/download/{GO2RTC_VERSION}"

# (system, machine) -> (asset filename, is_zip)
# Windows asset names confirmed directly against the v1.9.14 release's expanded asset
# list (not guessed) -- note the naming break from the darwin/linux entries: Windows
# uses "win64"/"win32"/"win_arm64", NOT "win_amd64". platform.machine() on 64-bit
# Windows returns "AMD64" (lowercased here to "amd64"), never "x86_64".
_ASSET_MAP: dict[tuple[str, str], tuple[str, bool]] = {
    ("darwin", "arm64"): ("go2rtc_mac_arm64.zip", True),
    ("darwin", "x86_64"): ("go2rtc_mac_amd64.zip", True),
    ("linux", "x86_64"): ("go2rtc_linux_amd64", False),
    ("linux", "aarch64"): ("go2rtc_linux_arm64", False),
    ("linux", "arm64"): ("go2rtc_linux_arm64", False),
    ("windows", "amd64"): ("go2rtc_win64.zip", True),
    ("windows", "x86"): ("go2rtc_win32.zip", True),
    ("windows", "arm64"): ("go2rtc_win_arm64.zip", True),
}


class UnsupportedPlatformError(RuntimeError):
    pass


def _resolve_asset() -> tuple[str, bool]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    key = (system, machine)
    if key not in _ASSET_MAP:
        raise UnsupportedPlatformError(
            f"no known go2rtc release asset for platform {system}/{machine}; "
            f"supported: {sorted(_ASSET_MAP)}"
        )
    return _ASSET_MAP[key]


def _binary_path(bin_dir: str) -> Path:
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    return Path(bin_dir) / f"go2rtc{suffix}"


def ensure_go2rtc_binary(bin_dir: str = GO2RTC_BIN_DIR) -> str:
    """Returns the path to a working go2rtc binary, downloading + extracting it into
    bin_dir if not already cached there. Idempotent: a second call with an
    already-cached binary does no network I/O.
    """
    dest = _binary_path(bin_dir)
    if dest.exists():
        return str(dest)

    Path(bin_dir).mkdir(parents=True, exist_ok=True)
    asset_name, is_zip = _resolve_asset()
    url = f"{GO2RTC_RELEASE_BASE}/{asset_name}"
    logger.info("downloading go2rtc %s from %s", GO2RTC_VERSION, url)

    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = resp.read()

    if is_zip:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(names) != 1:
                raise RuntimeError(f"expected exactly one file in go2rtc zip, found: {names}")
            with zf.open(names[0]) as member:
                dest.write_bytes(member.read())
    else:
        dest.write_bytes(payload)

    dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    logger.info("go2rtc binary ready at %s", dest)
    return str(dest)
