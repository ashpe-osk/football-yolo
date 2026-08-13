from ultralytics import YOLO
from pathlib import Path
from collections import Counter

print("=" * 60)
print("🔬 TESTING YOLO26 ON YOUR VIDEO (CPU)")
print("=" * 60)

# 1. Load your model
model_path = r"C:\football-yolo\models\best.pt"

if not Path(model_path).exists():
    print(f"❌ Model not found at: {model_path}")
    exit()

print(f"✅ Loading model from: {model_path}")
model = YOLO(model_path)
print(f"📋 Classes: {list(model.names.values())}")
print("=" * 60)

# 2. Your video
video_path = r"C:\football-yolo\data\videos\08fd33_4.mp4"

if not Path(video_path).exists():
    print(f"❌ Video not found: {video_path}")
    exit()

print(f"✅ Video: {Path(video_path).name}")
print("=" * 60)

# 3. Run tracking with CPU
print("\n🔍 Running tracking on CPU...")
print("   Resolution: 640x640")
print("   Confidence: 0.3")
print("   Device: CPU")
print("   Output: ./yolo26_output/")
print("=" * 60)

results = model.track(
    source=video_path,
    conf=0.3,
    iou=0.5,
    imgsz=640,
    persist=True,
    tracker="botsort.yaml",
    save=True,
    project="yolo26_output",
    name="test_run",
    exist_ok=True,
    device="cpu",  # CHANGED: Use CPU instead of GPU
    verbose=True,
)

print("\n" + "=" * 60)
print("✅ COMPLETE!")
print(f"📁 Output: yolo26_output/test_run/")
print("=" * 60)

# 4. Quick stats
print("\n📊 Detection Summary:")
labels_dir = Path("yolo26_output/test_run/labels")

if labels_dir.exists():
    counts = Counter()
    total = 0
    
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counts[cls_id] += 1
                    total += 1
    
    if total > 0:
        print(f"   Total detections: {total}")
        print("\n   Per class:")
        for cls_id, count in sorted(counts.items()):
            name = model.names.get(cls_id, f"Class {cls_id}")
            pct = (count / total) * 100
            print(f"   • {name:12}: {count:6} ({pct:5.1f}%)")
    else:
        print("   ⚠️ No detections found")
else:
    print("   ℹ️ Label files not saved")

print("\n" + "=" * 60)
print("🎯 Check the video: yolo26_output/test_run/")
print("   Look for the referee label consistency")
print("=" * 60)