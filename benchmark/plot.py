import json
import matplotlib.pyplot as plt
from collections import defaultdict

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Time unit conversion to ms
time_unit_to_ms = {
    'ns': 1e-6,
    'us': 1e-3,
    'ms': 1,
    's': 1e3
}

# Load the JSON file
with open('results2.json') as f:
    data = json.load(f)

# Filter relevant entries
median_entries = [
    b for b in data['benchmarks']
    if b.get('run_type') == 'aggregate' and b.get('aggregate_name') == 'median'
]

results = defaultdict(list)

for entry in median_entries:
    name = entry['run_name']
    parts = name.split('/')
    numeric_parts = [int(p) for p in parts if is_integer_string(p)]

    if len(numeric_parts) < 10:
        continue  # Malformed entry

    A_dims = tuple(numeric_parts[2:4])
    B_dims = tuple(numeric_parts[4:6])
    C_dims = tuple(numeric_parts[6:8])
    tile_size = tuple(numeric_parts[8:10])

    time_unit = entry.get('time_unit', 'ns')
    conversion = time_unit_to_ms.get(time_unit, 1e-6)  # fallback to ns->ms
    cpu_time_ms = entry['cpu_time'] * conversion

    key = (A_dims, B_dims, C_dims)
    results[key].append((tile_size, cpu_time_ms, time_unit))

# Plotting
plt.figure(figsize=(12, 7))
all_tile_sizes = list(set(tile_size for dims, runs in results.items() for (tile_size, _, _) in runs))
all_tile_sizes.sort()

tile_size_to_idx = {tile_size: i for i, tile_size in enumerate(all_tile_sizes)}

print(tile_size_to_idx)

all_tile_labels = [f"{tile_size[0]}x{tile_size[1]}" for tile_size in tile_size_to_idx.keys()]


for dims, runs in results.items():
    # if dims[0][0] == 2048 and dims[0][1] == 2048:
        # continue
#     runs.sort(key=lambda x: x[0])
#     tile_labels = [f"{tile_size[0]}x{tile_size[1]}" for (tile_size, _, _) in runs]
    cpu_times = [t for (_, t, _) in runs]
    x_vals = [tile_size_to_idx[tile_size] for (tile_size, _, _) in runs]
#     x_vals = list(range(len(tile_labels)))  # numeric x-axis for plotting
    label = f"A{dims[0]}, B{dims[1]}, C{dims[2]}"

    plt.plot(x_vals, cpu_times, marker='o', label=label)

    for i, (tile_size, t, _) in enumerate(runs):
        plt.annotate(f"{t:.1f} ms", (tile_size_to_idx[tile_size], t), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# # Use string labels as x-tick labels
plt.xticks(ticks=range(len(all_tile_labels)), labels=all_tile_labels, rotation=90)

plt.xlabel("Tile Size (HxW)")
plt.ylabel("CPU Time (ms)")
plt.title("Median CPU Time vs Tile Size for Different Matrix Multiplications")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_plot_full.png", dpi=300, bbox_inches='tight')
plt.show()