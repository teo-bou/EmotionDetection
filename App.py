import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # Move to the GPU

classes = ['Happy', 'Sad', 'Neutral']


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*6*6, 512)
        self.fc2 = nn.Linear(512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128*6*6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load('CNN_model.pth'))
model.to(device)
model.eval()
# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect faces
        results = face_detection.process(rgb_frame)

        # Draw face detections
        if results.detections:
            try:
                detection = results.detections[0]
                mp_drawing.draw_detection(frame, detection)
                xmin, ymin, width, height = int(detection.location_data.relative_bounding_box.xmin * frame.shape[1]), int(detection.location_data.relative_bounding_box.ymin * frame.shape[0]), int(detection.location_data.relative_bounding_box.width * frame.shape[1]), int(detection.location_data.relative_bounding_box.height * frame.shape[0])
                face = frame[ymin:ymin+height, xmin:xmin+width]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = torch.tensor(face, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                face = face.to(device)
                prediction = model(face)
                prediction = F.softmax(prediction, dim=1)
                prediction = torch.argmax(prediction, dim=1)
                cv2.putText(frame, f'Emotion: {classes[prediction.item()]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except cv2.error as e:
                print(f'Error {e}')
        
        # Display the frame
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()