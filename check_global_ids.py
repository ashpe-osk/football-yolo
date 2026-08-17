import pickle
import sys
import os

if len(sys.argv) > 1:
    stub_path = sys.argv[1]
else:
    # If no argument, ask interactively
    stub_path = input("Enter the path to your tracks pickle file: ").strip()
    if not os.path.exists(stub_path):
        print(f"File not found: {stub_path}")
        sys.exit(1)

with open(stub_path, 'rb') as f:
    tracks = pickle.load(f)

all_gids = set()
max_per_frame = 0

for frame_players in tracks['players']:
    gids = set(frame_players.keys())
    all_gids.update(gids)
    max_per_frame = max(max_per_frame, len(gids))

print(f"Maximum players visible in a single frame: {max_per_frame}")
print(f"Total distinct Global IDs ever created: {len(all_gids)}")