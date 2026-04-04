import importlib
import time
start = time.perf_counter()

#=====
RUNS = {
    "hcn": "runs.run_hcn",
    "cocaine2": "runs.run_cocaine2",
    "asp_decarb": "runs.run_aspartate_decarb",
}

def main(run_name):
    modname = RUNS.get(run_name)
    mod = importlib.import_module(modname)
    return 0

if __name__ == "__main__":
    main("asp_decarb")


#=====timer tail
for _ in range(1_000_000):
    pass
end = time.perf_counter()
elapsed_us = (end - start)
print(f"\nTime wasted: {elapsed_us:.2f} s")

#=====finish alarm
import winsound
winsound.Beep(600, 700)