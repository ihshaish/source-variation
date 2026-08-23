"""Download Peter's refreshed ASRS corpus (Drive folder 'NASA ASRS', 2026-08-03)
into data_asrs_v2/. Files are link-shared; direct uc?export=download works."""
import json, os, subprocess, sys, time

FILES = {}  # id -> filename, deduped
listing = json.load(open(os.path.join(os.path.dirname(__file__), "v2_manifest.json")))
for f in listing:
    FILES[f["id"]] = f["title"]

out_dir = "/Users/Hisham/github_page/PhD_peter/data_asrs_v2"
ok = fail = skip = 0
for fid, name in sorted(FILES.items(), key=lambda kv: kv[1]):
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        skip += 1
        continue
    r = subprocess.run(["curl", "-sL", "--fail", "--retry", "3",
                        f"https://drive.google.com/uc?export=download&id={fid}",
                        "-o", dest], capture_output=True)
    # basic sanity: DBOL CSVs start with a section header row containing ' Time'
    good = r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 100_000
    if good:
        with open(dest, errors="replace") as fh:
            good = "Time" in fh.readline()
    time.sleep(3)
    if good:
        ok += 1
        print(f"ok   {name}")
    else:
        fail += 1
        print(f"FAIL {name}")
        if os.path.exists(dest):
            os.remove(dest)
print(f"done: {ok} ok, {skip} skipped, {fail} failed of {len(FILES)}")
