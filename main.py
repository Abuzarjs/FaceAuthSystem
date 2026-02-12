import cv2
import pandas as pd
from datetime import datetime
from face_logic import FaceEngine

# Initialize Attendance File if not exists
try:
    df = pd.read_csv('attendance.csv')
except:
    df = pd.DataFrame(columns=['Name', 'Date', 'Punch-In', 'Punch-Out'])
    df.to_csv('attendance.csv', index=False)

engine = FaceEngine()
video_capture = cv2.VideoCapture(0)

def mark_attendance(name):
    global df
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    # Logic: If name is in CSV for today, update Punch-Out, else create Punch-In
    mask = (df['Name'] == name) & (df['Date'] == date_str)
    if not df[mask].any().any():
        new_record = {'Name': name, 'Date': date_str, 'Punch-In': time_str, 'Punch-Out': ''}
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        df.loc[mask, 'Punch-Out'] = time_str
    
    df.to_csv('attendance.csv', index=False)

print("System Started. Press 'q' to quit, 'p' to punch attendance.")

while True:
    ret, frame = video_capture.read()
    locations, names = engine.recognize_face(frame)

    for (top, right, bottom, left), name in zip(locations, names):
        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Trigger punch with 'p' key when a face is detected
        if cv2.waitKey(1) & 0xFF == ord('p') and name != "Unknown":
            mark_attendance(name)
            print(f"Attendance marked for {name}")

    cv2.imshow('Face Authentication Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()