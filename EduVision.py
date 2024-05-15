import os
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


def detect_persons(video_path, output_dir):
    """Detect persons in the video and save cropped images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(frame_rate / 5)
    frame_id = 0
    person_count = 0
    margin = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frames == 0:
            input_tensor = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(input_tensor)

            for element in range(len(prediction[0]['boxes'])):
                box = prediction[0]['boxes'][element].cpu().numpy()
                score = prediction[0]['scores'][element].cpu().numpy()
                label_idx = prediction[0]['labels'][element].cpu().numpy()
                if score > 0.5 and label_idx == 1:
                    x1, y1, x2, y2 = map(int, box)
                    x1 -= margin
                    y1 -= margin
                    x2 += margin
                    y2 += margin
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    cropped_image = frame[y1:y2, x1:x2]
                    person_count += 1
                    image_path = os.path.join(output_dir, f'person_frame_{frame_id}_{person_count}.jpg')
                    cv2.imwrite(image_path, cropped_image)

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()



def classify_face_orientations(input_dir, output_dir):
    """Classify face orientations."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    class_counts = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            image_drawn = image.copy()

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    img_h, img_w, _ = image.shape
                    face_2d = []
                    face_3d = []
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x, y = int(landmark.x * img_w), int(landmark.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, landmark.z])

                    if face_2d and face_3d:
                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)
                        focal_length = img_w
                        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                               [0, focal_length, img_h / 2],
                                               [0, 0, 1]], dtype='double')
                        dist_matrix = np.zeros((4, 1), dtype=np.float64)
                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        rmat, jac = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

                        if y < -8:
                            text = "Looking_Left"
                        elif y > 8:
                            text = "Looking_Right"
                        elif x < -8:
                            text = "Looking_Down"
                        elif x > 8:
                            text = "Looking_Up"
                        else:
                            text = "Looking_Forward"

                        class_directory = os.path.join(output_dir, text)
                        if not os.path.exists(class_directory):
                            os.makedirs(class_directory)

                        new_filename = f"{text}_{filename}"
                        save_path = os.path.join(class_directory, new_filename)
                        mp_drawing.draw_landmarks(
                            image=image_drawn,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                        cv2.imwrite(save_path, image_drawn)
                        class_counts[text] = class_counts.get(text, 0) + 1

    cv2.destroyAllWindows()
    return class_counts










def detect_hand_raised(input_dir, output_dir):
    """Detect hand raised."""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    hand_raised_counts = 0

    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        left_stage, right_stage = None, None
        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                try:
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    image_height, image_width, _ = image.shape
                    left_wrist_pixel = (int(left_wrist[0] * image_width), int(left_wrist[1] * image_height))
                    right_wrist_pixel = (int(right_wrist[0] * image_width), int(right_wrist[1] * image_height))

                    image_with_skeleton = image.copy()
                    mp_drawing.draw_landmarks(image_with_skeleton, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(image_with_skeleton, f'Left Angle: {left_angle:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image_with_skeleton, f'Right Angle: {right_angle:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image_with_skeleton, f'Left Wrist: {left_wrist_pixel}', left_wrist_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image_with_skeleton, f'Right Wrist: {right_wrist_pixel}', right_wrist_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                    if left_wrist[1] < 0.3 and left_stage == 'down':
                        left_stage = "up"
                        output_path = os.path.join(output_dir, filename)
                        cv2.imwrite(output_path, image_with_skeleton)
                        hand_raised_counts += 1
                    elif left_angle > 160:
                        left_stage = "down"
                    if right_wrist[1] < 0.3 and right_stage == 'down':
                        right_stage = "up"
                        output_path = os.path.join(output_dir, filename)
                        cv2.imwrite(output_path, image_with_skeleton)
                        hand_raised_counts += 1
                    elif right_angle > 160:
                        right_stage = "down"
                except:
                    pass
    return hand_raised_counts


def detect_phones(input_dir, output_dir):
    """Detect phones."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()
    class_mapping = {77: "Phone"}
    phone_counts = {}

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(class_path, filename)
                    input_image = to_tensor(cv2.imread(image_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        prediction = model(input_image)
                    original_image = cv2.imread(image_path)
                    detected_objects = []

                    for element in range(len(prediction[0]['boxes'])):
                        box = prediction[0]['boxes'][element].cpu().numpy()
                        score = prediction[0]['scores'][element].cpu().numpy()
                        label_idx = prediction[0]['labels'][element].cpu().item()

                        if label_idx in class_mapping and score > 0.5:
                            detected_objects.append(class_mapping[label_idx])
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(original_image, f'{class_mapping[label_idx]}: {score:.2f}',
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if detected_objects:
                        detected_objects_str = "_".join(sorted(set(detected_objects)))
                        output_directory = os.path.join(output_dir, f"{class_name}_Detected_{detected_objects_str}")
                        if not os.path.exists(output_directory):
                            os.makedirs(output_directory)
                        new_filename = f"{class_name}_{detected_objects_str}_{filename}"
                        output_path = os.path.join(output_directory, new_filename)
                        cv2.imwrite(output_path, original_image)
                        phone_counts[new_filename] = phone_counts.get(new_filename, 0) + 1

    return phone_counts


import os
import cv2
import numpy as np
import pickle
import pandas as pd
import mediapipe as mp

def Sleeping_Pose(input_dir, output_dir, model):
    """Process images from a directory and save specific class outputs."""
    # Load body language model
    with open(model, 'rb') as f:
        model = pickle.load(f)
    
    # Ensure output directory exists
    sleeping_pose_dir = os.path.join(output_dir, 'Sleeping_Pose')
    if not os.path.exists(sleeping_pose_dir):
        os.makedirs(sleeping_pose_dir)
    
    # Initiate holistic model
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Process image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            # Extract data for model prediction
            try:
                # Assume landmarks extraction similar to your provided structure
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                row = pose_row + face_row
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                
                # Save only 'Sleeping_Pose' class images
                if body_language_class == 'Sleeping_Pose':
                    save_path = os.path.join(sleeping_pose_dir, filename)
                    cv2.imwrite(save_path, image)
                    print(f"Saved {body_language_class} at {save_path}")
                
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
    
    holistic.close()




import os
import matplotlib.pyplot as plt

def count_files(directory):
    """Count files in the given directory, assuming they're images."""
    if not os.path.exists(directory):
        return 0
    return len([name for name in os.listdir(directory) if name.endswith(('.png', '.jpg', '.jpeg'))])

def plot_final_chart(counts_dict):
    path_classified = 'Classified_Images'
    path_detected = 'Classified_Images_Detected'
    path_hand_raised = os.path.join(path_classified, 'Hand_Raised')
    
    class_names = ['Looking_Down', 'Looking_Forward', 'Looking_Left', 'Looking_Right', 'Looking_Up']
    
    classified_counts = {class_name: count_files(os.path.join(path_classified, class_name))
                         for class_name in class_names}
    detected_counts = {class_name + '_Detected_Phone': count_files(os.path.join(path_detected, class_name + '_Detected_Phone'))
                       for class_name in class_names}

    # --- First figure ---

    categories = class_names
    total_classified = [classified_counts.get(class_name, 0) for class_name in class_names]
    total_detected = [detected_counts.get(class_name + '_Detected_Phone', 0) for class_name in class_names]

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    index = range(len(categories))

    rects1 = ax.bar(index, total_classified, bar_width, color='royalblue', label='Classified Images')
    rects2 = ax.bar([p + bar_width for p in index], total_detected, bar_width, color='seagreen', label='Detected Phone')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_title('EduVision - Figure 1')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(categories)
    ax.legend()

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)

    plt.tight_layout()
    plt.show()

    # --- Second figure ---

    hand_raised_classified = count_files(path_hand_raised)
    hand_raised_detected = 0  # Assuming no detected hand raised

    distracted_classified = sum([classified_counts.get(class_name, 0) for class_name in ['Looking_Up', 'Looking_Down', 'Looking_Left', 'Looking_Right']])
    distracted_detected = sum([detected_counts.get(class_name + '_Detected_Phone', 0) for class_name in ['Looking_Up', 'Looking_Down', 'Looking_Left', 'Looking_Right']])
    
    # Count every class that has detected phone, not just Looking Down
    distracted_with_phone = sum([detected_counts.get(class_name + '_Detected_Phone', 0) for class_name in class_names])

    focused_classified = classified_counts.get('Looking_Forward', 0)
    focused_detected = detected_counts.get('Looking_Forward_Detected_Phone', 0)

    categories = ['Distracted', 'Distracted with Phone', 'Focused', 'Hand Raised']
    totals_classified = [distracted_classified, distracted_with_phone, focused_classified, hand_raised_classified]
    totals_detected = [distracted_detected, distracted_with_phone, focused_detected, hand_raised_detected]

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(range(len(categories)), totals_classified, bar_width, color='royalblue', label='Classified Images')
    

    ax.set_xlabel('Categories')
    ax.set_ylabel('Counts')
    ax.set_title('EduVision - Figure 2')
    ax.set_xticks([p + bar_width / 2 for p in range(len(categories))])
    ax.set_xticklabels(categories)
    ax.legend()

    add_labels(rects1)
    #add_labels(rects2)

    plt.tight_layout()
    plt.show()






def main_video_processing(video_path):
    output_dir = 'Output_Segmentation'
    classified_dir = 'Classified_Images'
    hand_raised_dir = os.path.join(classified_dir, 'Hand_Raised')
    detected_dir = 'Classified_Images_Detected'
    model_path = 'body_language.pkl'
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(classified_dir):
        os.makedirs(classified_dir)
    if not os.path.exists(hand_raised_dir):
        os.makedirs(hand_raised_dir)
    if not os.path.exists(detected_dir):
        os.makedirs(detected_dir)

    detect_persons(video_path, output_dir)
    face_counts = classify_face_orientations(output_dir, classified_dir)
    hand_raised_counts = detect_hand_raised(output_dir, hand_raised_dir)
    phone_counts = detect_phones(classified_dir, detected_dir)
    #Sleeping_Pose(output_dir, classified_dir, model_path)

    final_counts = {**face_counts, 'Hand_Raised': hand_raised_counts, **phone_counts}
    plot_final_chart(final_counts)


if __name__ == '__main__':
    video_path = 'Test f5.mp4'
    main_video_processing(video_path)

    