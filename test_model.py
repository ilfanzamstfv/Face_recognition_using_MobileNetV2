import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

# --- Konfigurasi ---
MODEL_H5_FILE = "model/face_classifier_model.h5" # sesuaikan dengan penyimpanan model.h5
LABEL_MAP_FILE = "model/class_indices.json" # sesuaikan dengan penyimpanan class.json
FACE_DETECTION_MODEL = "hog"  # atau "cnn" untuk akurasi lebih tinggi (lebih lambat)
CONFIDENCE_THRESHOLD = 0.8  # Ambang batas kepercayaan prediksi
FRAME_RESIZE = 0.25  # Resize frame untuk proses lebih cepat
FONT = cv2.FONT_HERSHEY_DUPLEX

def load_model_and_labels():
    """Memuat model dan label mapping"""
    if not os.path.exists(MODEL_H5_FILE) or not os.path.exists(LABEL_MAP_FILE):
        raise FileNotFoundError("File model atau label map tidak ditemukan!")
    
    # Muat model Keras
    model = keras.models.load_model(MODEL_H5_FILE)
    print("Model berhasil dimuat")
    
    # Muat label mapping
    with open(LABEL_MAP_FILE, 'r') as f:
        label_map = json.load(f)
    
    # Buat mapping dari int ke string (kebalikan dari yang disimpan)
    int_to_label = {v: k for k, v in label_map.items()}
    print("Label mapping berhasil dimuat")
    
    return model, int_to_label

def recognize_faces(frame, model, int_to_label):
    """Mengenali wajah dalam frame dan memberikan label"""
    # Ubah ukuran frame untuk proses lebih cepat
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    
    # Konversi BGR (OpenCV) ke RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi lokasi wajah
    face_locations = face_recognition.face_locations(rgb_small_frame, model=FACE_DETECTION_MODEL)
    
    # Jika tidak ada wajah terdeteksi
    if not face_locations:
        return frame, []
    
    # Ekstrak encoding wajah
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # Lakukan prediksi
    predictions = model.predict(np.array(face_encodings))
    
    # Proses hasil prediksi
    recognized_faces = []
    for i, pred in enumerate(predictions):
        # Dapatkan label dan confidence score
        pred_label = np.argmax(pred)
        confidence = np.max(pred)
        
        # Jika confidence di bawah threshold, beri label "Unknown"
        if confidence < CONFIDENCE_THRESHOLD:
            label = "Unknown"
        else:
            label = int_to_label.get(pred_label, "Unknown")
        
        recognized_faces.append({
            "location": face_locations[i],
            "label": label,
            "confidence": confidence
        })
    
    return frame, recognized_faces

def draw_face_annotations(frame, faces_info, frame_resize=FRAME_RESIZE):
    """Menggambar kotak dan label pada wajah yang terdeteksi"""
    for face in faces_info:
        # Skala kembali lokasi wajah ke ukuran frame asli
        top, right, bottom, left = face["location"]
        top = int(top / frame_resize)
        right = int(right / frame_resize)
        bottom = int(bottom / frame_resize)
        left = int(left / frame_resize)
        
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Gambar label dengan background
        label = f"{face['label']} ({face['confidence']:.2f})"
        label_size = cv2.getTextSize(label, FONT, 0.5, 1)
        
        # Rectangle untuk background text
        cv2.rectangle(frame, (left, bottom - label_size[0][1] - 10), 
                     (left + label_size[0][0], bottom), (0, 255, 0), cv2.FILLED)
        
        # Text label
        cv2.putText(frame, label, (left, bottom - 10), FONT, 0.5, (0, 0, 0), 1)
    
    return frame

def main():
    print("Memulai pengenalan wajah real-time...")
    
    try:
        # Muat model dan label
        model, int_to_label = load_model_and_labels()
        
        # Buka webcam
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise RuntimeError("Tidak dapat mengakses webcam!")
        
        print("Tekan 'q' untuk keluar...")
        
        while True:
            # Ambil frame dari webcam
            ret, frame = video_capture.read()
            if not ret:
                print("Tidak dapat menerima frame dari webcam")
                break
            
            # Kenali wajah dalam frame
            processed_frame, faces_info = recognize_faces(frame, model, int_to_label)
            
            # Gambar hasil deteksi
            output_frame = draw_face_annotations(processed_frame, faces_info)
            
            # Tampilkan hasil
            cv2.imshow('Face Recognition', output_frame)
            
            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Bersihkan
        video_capture.release()
        cv2.destroyAllWindows()
        print("Program dihentikan")
    
    except Exception as e:
        print(f"Error: {e}")
        if 'video_capture' in locals():
            video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
