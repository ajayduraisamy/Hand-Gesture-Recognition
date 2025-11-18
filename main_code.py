# =====================================================
# gesture_tts.py 
# =====================================================

import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import copy
import itertools
import time
import threading
import string
from tensorflow.keras.models import load_model
from gtts import gTTS
from playsound import playsound
from utils import CvFpsCalc
import speech_recognition as sr

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "trained_model_CNN.h5"
GESTURE_IMAGE_PATH = "../gesture_images"
TTS_CACHE_DIR = "tts_cache"
STABLE_FRAME_THRESHOLD = 4
SPEAK_COOLDOWN = 3.0
PROB_THRESHOLD = 0.75  # Ignore gestures with low confidence

if not os.path.isdir(TTS_CACHE_DIR):
    os.makedirs(TTS_CACHE_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

results_text1 = [
    '', 'hi', 'hello', 'fine', 'how are you', 'How its going?', 'Nice to see you!',
    'How you been?', 'Lovely to meet you', 'Its a pleasure to meet you',
    'Have a great day!', 'See you tomorrow', 'Talk to you later', 'I missed you!',
    'You look well!', 'Congratulations!', 'All the best!', 'Happy Birthday!',
    'Hello there!', 'Take care!', 'Catch you later!', 'See you around!', 'Bye!',
    'Good morning', 'Good night', 'Yo buddy!', 'You made my day!'
]

# =====================================================
# AUDIO HELPERS
# =====================================================
def play_audio_file(path):
    try:
        playsound(path)
    except Exception as e:
        print("Audio play error:", e)


def speak_text_cached(text):
    if not text:
        return
    safe_name = "".join(c for c in text if c.isalnum() or c in (' ', '_')).rstrip()
    filename = os.path.join(TTS_CACHE_DIR, f"{safe_name[:80].replace(' ', '_')}.mp3")
    if not os.path.exists(filename):
        try:
            tts = gTTS(text)
            tts.save(filename)
        except Exception as e:
            print("TTS generation error:", e)
            return
    threading.Thread(target=play_audio_file, args=(filename,), daemon=True).start()

# =====================================================
# MEDIA PIPE & LANDMARK HELPERS
# =====================================================
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_array = np.append(landmark_array, [[landmark_x, landmark_y]], axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for i, point in enumerate(temp_landmark_list):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list)) if temp_landmark_list else 1
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

# =====================================================
# GESTURE → VOICE MODE 
# =====================================================
def main_gesture_to_voice():
    print("👉 Gesture to Voice Mode Activated! (Press 'c' to change mode)\n")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    prev_label = None
    stable_count = 0
    last_spoken_time = 0.0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(1) & 0xFF
        if key == ord('c'):  # Switch mode
            print(" Changing mode...")
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                processed = pre_process_landmark(landmark_list)
                if len(processed) != 42:
                    continue

                pred = model.predict(np.array(processed).reshape(1, 42))
                label_idx = int(np.argmax(pred))
                prob = float(np.max(pred))

                if prob < PROB_THRESHOLD:
                    prev_label = None
                    stable_count = 0
                    continue

                if label_idx >= len(results_text1) or not results_text1[label_idx].strip():
                    prev_label = None
                    stable_count = 0
                    continue

                if prev_label is None or label_idx != prev_label:
                    prev_label = label_idx
                    stable_count = 1
                else:
                    stable_count += 1

                now = time.time()
                can_speak = (stable_count >= STABLE_FRAME_THRESHOLD) and (now - last_spoken_time >= SPEAK_COOLDOWN)
                spoken_text = results_text1[label_idx]

                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
                cv.putText(debug_image, spoken_text, (brect[0], brect[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if can_speak and spoken_text:
                    speak_text_cached(spoken_text)
                    last_spoken_time = now
                    stable_count = 0

        cv.putText(debug_image, f"FPS: {fps}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Hand Gesture → Voice', debug_image)

    cap.release()
    cv.destroyAllWindows()

# =====================================================
# VOICE → GESTURE MODE 
# =====================================================
def voice_to_gesture():
    print("\n🎤 Voice to Gesture Mode Activated! (Press 'c' to change mode)\n")

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        print("Say a greeting (e.g., 'hello', 'fine', 'good morning') or press 'c' to change mode...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

        # Check for mode change keypress
        if cv.waitKey(1) & 0xFF == ord('c'):
            print(" Changing mode...")
            break

        try:
            text = recognizer.recognize_google(audio).strip().lower()
            print(f" You said: {text}")

            normalized_text = text.translate(str.maketrans('', '', string.punctuation)).replace(" ", "")

            found = False
            for file in os.listdir(GESTURE_IMAGE_PATH):
                filename_no_ext = os.path.splitext(file)[0]
                clean_filename = filename_no_ext.lower().translate(str.maketrans('', '', string.punctuation)).replace(" ", "")
                if normalized_text == clean_filename:
                    img_path = os.path.join(GESTURE_IMAGE_PATH, file)
                    img = cv.imread(img_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        scale = 1.8
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        resized_img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)

                        window_name = f"Gesture for '{text}'"
                        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
                        cv.imshow(window_name, resized_img)

                        speak_text_cached(f"Showing gesture for {text}")
                        cv.waitKey(1000)
                        cv.destroyAllWindows()
                        found = True
                    break

            if not found:
                print(f" No gesture image found for: {text}")

        except sr.UnknownValueError:
            print(" Could not understand the speech.")
        except sr.RequestError:
            print(" Speech recognition service unavailable.")

# =====================================================
# MAIN MENU
# =====================================================
if __name__ == "__main__":
    while True:
        print("\n==============================")
        print("Press 0 → Gesture → Voice (Camera Mode)")
        print("Press 1 → Voice → Gesture (Speak Mode)")
        print("Press q → Quit")
        print("==============================\n")

        choice = input("Enter your choice: ").strip().lower()

        if choice == '0':
            main_gesture_to_voice()
        elif choice == '1':
            voice_to_gesture()
        elif choice == 'q':
            print(" Exiting program.")
            break
        else:
            print(" Invalid input. Try again.")
