"""Move the SFT parquet set through the bucket. `up` on worker 0 after the build, `down` on every worker.
Verifies each file's md5 against the bucket so a bad copy can't slip into training."""
import base64, glob, hashlib, os, sys
import gcsfs

SET = os.environ.get("SFT_SET", "kisoku-sft-v2")  # v1 = preview set, v2 = final set (built on the Mac, uploaded with gsutil)
REMOTE = f"kisoku-v2-training/sft/{SET}"
BUILD = os.path.expanduser(f"~/sft-build/{SET}")
LOCAL = os.path.expanduser(f"~/sft-data/{SET}")

fs = gcsfs.GCSFileSystem(token=os.path.expanduser("~/gcs-key.json"))


def md5_b64(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def remote_md5(rpath):
    return fs.info(rpath)["md5Hash"]


mode = sys.argv[1]
if mode == "up":
    files = sorted(glob.glob(f"{BUILD}/*"))
    for p in files:
        fs.put(p, f"{REMOTE}/{os.path.basename(p)}")
    bad = [p for p in files if md5_b64(p) != remote_md5(f"{REMOTE}/{os.path.basename(p)}")]
    print(f"uploaded {len(files)} files, md5 mismatches: {len(bad)}", bad)
elif mode == "down":
    os.makedirs(LOCAL, exist_ok=True)
    names = sorted(os.path.basename(r) for r in fs.ls(REMOTE))
    for n in names:
        fs.get(f"{REMOTE}/{n}", f"{LOCAL}/{n}")
    bad = [n for n in names if md5_b64(f"{LOCAL}/{n}") != remote_md5(f"{REMOTE}/{n}")]
    print(f"{os.uname().nodename}: downloaded {len(names)} files, md5 mismatches: {len(bad)}", bad)
