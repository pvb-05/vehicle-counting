from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2 as cv

# load model
model = YOLO("yolov8m.pt")

# load deepsort
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.3,
)

# load video
video = cv.VideoCapture("assets/video.mp4")

# declare videoWriter object to read
original_fps = video.get(cv.CAP_PROP_FPS)
original_w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
original_h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
out = cv.VideoWriter(
    filename="result.mp4", 
    fourcc=cv.VideoWriter_fourcc(*"mp4v"), 
    fps=original_fps, 
    frameSize=(original_w, original_h)
)

# config count line
LINE = int(original_h*0.6)
OFFSET = 40

# config counters
tracked = set()

# config classes
VEHICLES = {
    2: "Car",
    3: "Motor",
    5: "Bus",
    7: "Truck"
}

COUNTERS = {
    "Car": 0,
    "Motor": 0,
    "Bus": 0,
    "Truck": 0
}

def main():
    while True:
        # read a frame
        ret, frame = video.read()
        if not ret:
            break

        # dect objects
        results = model(
            frame,
            stream=True,
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls not in VEHICLES.keys() or conf < 0.5:
                    continue

                detections.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            track_class = track.det_class
            ltrb = track.to_ltrb()

            left = int(ltrb[0])
            top = int(ltrb[1])
            right = int(ltrb[2])
            bottom = int(ltrb[3])

            # draw line 
            cv.line(frame, (0, LINE), (original_w, LINE), (0, 0, 255), 2)

            cx = int((left+right)/2)
            cy = int((top+bottom)/2)

            if (LINE- OFFSET) < cy < (LINE + OFFSET):
              if track_id not in tracked:
                tracked.add(track_id)

                if track_class in VEHICLES.keys():
                    COUNTERS[VEHICLES[track_class]] +=1

            # draw bound box
            cv.rectangle(
                img=frame,
                pt1=(left,top),
                pt2=(right,bottom),
                color=(0, 255, 0),
                thickness=2
            )

            # draw class,id and center
            cv.putText(
                img=frame,
                text=f"ID: {track_id} Class:{VEHICLES.get(track_class)}",
                org=(left, top),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=2
            )

            cv.circle(
                img=frame,
                center=(cx,cy),
                radius=4,
                color=(0, 0, 255),
                thickness=2
            )
    

        cv.putText(
            img=frame, 
            text=f'Count: {len(tracked)}', 
            org=(30, 50), 
            fontFace=cv.FONT_HERSHEY_SIMPLEX, 
            fontScale=1.5, 
            color=(0, 0, 255), 
            thickness=3
        )

        y_pos = 60
        
        for name, count in COUNTERS.items():
          y_pos += 30 
          color = (255, 255, 255) 
          if name == "Car": color = (0, 255, 255)   
          if name == "Truck": color = (0, 165, 255)
          if name == "Bus": color = (255, 0, 255)   
                    
          cv.putText(
              img=frame,
              text=f"{name}: {count}", 
              org=(30, y_pos), 
              fontFace=cv.FONT_HERSHEY_SIMPLEX, 
              fontScale=0.7,
              color=color,
              thickness=2
          )

        cv.imshow("Vehicle Counting", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
          break

        out.write(frame)

    video.release()
    out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
