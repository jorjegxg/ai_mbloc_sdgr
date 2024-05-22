import cv2
import keyboard

threshold = 50

def initialize_model(config_file, frozen_model):
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    model.setInputSize(320, 320)
    model.setInputScale(1 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    return model

def load_class_labels(file_name):
    with open(file_name, 'rt') as fpt:
        return fpt.read().rstrip('\n').split('\n')

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    return cap

def mergiLaStanga():
    print("Mergi la stanga")
    keyboard.press_and_release('left')

def mergiLaDreapta():
    print("Mergi la dreapta")
    keyboard.press_and_release('right')

def mergiInFata():
    print("Mergi in față")
    keyboard.press_and_release('up')

def detect_objects(model, frame, classLabels):
    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    if len(classIndex) != 0:
        for ClassInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                class_name = classLabels[ClassInd - 1]
                x, y, width, height = boxes
                cv2.rectangle(frame, boxes, (32, 110, 27), 2)
                
                # Setare font și culoare text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                text_color = (255, 255, 255)  # Alb

                # Poziționare text
                text_x, text_y = boxes[0], boxes[1] - 10
                
                # Afișare text
                cv2.putText(frame, classLabels[ClassInd - 1], (text_x, text_y), font, font_scale, text_color, thickness=2)

                if class_name.lower() == "person":
                    frame_center = frame.shape[1] // 2
                    box_center = x + width // 2
                    if box_center < frame_center - threshold:
                        mergiLaStanga()
                    elif box_center > frame_center + threshold:
                        mergiLaDreapta()
                    else:
                        mergiInFata()

    return frame







def main():
    config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model = "frozen_inference_graph.pb"
    class_file = "coco.names"

    model = initialize_model(config_file, frozen_model)
    classLabels = load_class_labels(class_file)
    cap = initialize_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_objects(model, frame, classLabels)
        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
