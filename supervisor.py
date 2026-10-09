"""Passwords for protected actions. Two separate ones:
    supervisor  clear the inbox, save the relay mapping
    admin       AUTO LEARNING switch and its target

No password is written in the code, the config or a log. The first time a protected action is used, the
password is set in the program; only a salted hash is stored in machine_config.json (which is not in git).
"""
import hashlib
import hmac
import json
import os

KEY = "supervisor_password"
ADMIN_KEY = "admin_password"


def _read(path) -> dict:
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()


def is_set(path, key=KEY) -> bool:
    return isinstance(_read(path).get(key), dict)


def set_password(path, password: str, key=KEY):
    if len(password.strip()) < 4:
        raise ValueError("The password needs at least 4 characters.")
    data = _read(path)
    salt = os.urandom(16)
    data[key] = {"salt": salt.hex(), "hash": _hash(password.strip(), salt)}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, str(path))


def check(path, password: str, key=KEY) -> bool:
    stored = _read(path).get(key)
    if not isinstance(stored, dict) or password is None:
        return False
    try:
        return hmac.compare_digest(_hash(password.strip(), bytes.fromhex(stored["salt"])), stored["hash"])
    except (KeyError, ValueError):
        return False
