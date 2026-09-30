from ultralytics import YOLO


def main():
    model = YOLO("models/best.pt")

    results = model.predict("data/videos/vid_for_data_30s.mp4", save=True)
    print(results[0])
    print("-----------------------------")
    for box in results[0].boxes:
        print(box)


if __name__ == "__main__":
    main()
