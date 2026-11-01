import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.multihop_nav_env import MultihopNavEnv

DATA_PATH = "data/level1_fixed.json"


def ensure_data():
    if not Path(DATA_PATH).exists():
        print("Data not found, generating...")
        subprocess.run(
            [sys.executable, "-m", "data.multihop_generator",
             "--n", "200", "--out", DATA_PATH],
            check=True,
        )


def main():
    ensure_data()
    env = MultihopNavEnv(DATA_PATH)

    all_pass = True
    for i in range(10):
        obs, _ = env.reset()
        total = 0.0
        info = {}
        for a in [0, 1, 2, 3]:
            obs, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        hop_status = info["hop_status"]
        ok = (hop_status == [1, 1, 1]) and (abs(total - 5.6) < 0.01)
        flag = "OK  " if ok else "FAIL"
        print(f"[{flag}] ep#{i}  total={total:+.2f}  hop_status={hop_status}")
        if not ok:
            all_pass = False

    print("\n" + ("All 10 episodes passed." if all_pass else "Some episodes failed."))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
