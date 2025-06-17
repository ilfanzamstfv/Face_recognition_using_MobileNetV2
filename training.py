import face_recognition
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image

# --- Konfigurasi Pelatihan ---
DATASET_PATH = "dataset_train_friends" #penamaan dataset sesuai folder
MODEL_H5_FILE = "face_classifier_model.h5" #penamaan model
LABEL_MAP_FILE = "face_labels.json"
IMG_SIZE = 224  # Ukuran input MobileNetV2
FACE_DETECTION_MODEL = "hog"
LEARNING_RATE = 0.001  # Lebih kecil untuk transfer learning, dapat diatur sendiri
EPOCHS = 80 # dapat diatur sendiri
BATCH_SIZE = 16 # dapat diatur sendiri

def extract_and_preprocess_faces(dataset_path):
    """Ekstrak wajah dan preprocess untuk MobileNetV2"""
    face_images = []
    face_names = []
    
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir): continue
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            try:
                image = face_recognition.load_image_file(img_path)
                face_locations = face_recognition.face_locations(image, model=FACE_DETECTION_MODEL)
                
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    face_image = image[top:bottom, left:right]
                    
                    # Preprocessing khusus MobileNetV2
                    pil_img = Image.fromarray(face_image).resize((IMG_SIZE, IMG_SIZE))
                    img_array = np.array(pil_img)
                    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
                    
                    face_images.append(img_array)
                    face_names.append(person_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return np.array(face_images), face_names

def build_mobilenet_model(num_classes):
    """Bangun model berbasis MobileNetV2"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_keras_model(X, y, num_classes, model_path):
    """Latih model dengan validasi"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model, base_model = build_mobilenet_model(num_classes)
    model.summary()
    
    # Augmentasi
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    return model, history

def fine_tune_model(model, base_model, X_train, y_train, X_val, y_val):
    """Fine-tuning dengan unfreeze beberapa layer"""
    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile dengan learning rate lebih rendah
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # --- Data Augmentation ---
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Latih ulang
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=int(EPOCHS/2),
        batch_size=BATCH_SIZE
    )
    
    return model, history

def plot_training_history(history):
    """Fungsi untuk memvisualisasikan kurva akurasi dan loss selama training"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')  # Simpan plot sebagai gambar
    plt.show()

def plot_confusion_matrix(model, X_val, y_val, class_names):
    """Fungsi untuk membuat dan menampilkan confusion matrix"""
    print("\nMembuat Confusion Matrix...")
    
    # Prediksi pada data validasi
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Buat confusion matrix
    cm = confusion_matrix(y_val, y_pred_classes)
    
    # Visualisasi dengan seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')  # Simpan sebagai gambar
    plt.show()
    
    # Hitung dan tampilkan akurasi per kelas
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\nAkurasi per kelas:")
    for i, acc in enumerate(class_accuracy):
        print(f"{class_names[i]}: {acc:.2f}")

if __name__ == "__main__":
    
    try:
        # 1. Ekstrak dan preprocess gambar wajah
        face_images, face_names = extract_and_preprocess_faces(DATASET_PATH)
        
        # 2. Encode labels
        label_encoder = LabelEncoder()
        labels_int = label_encoder.fit_transform(face_names)
        num_classes = len(label_encoder.classes_)
        
        # 3. Simpan label mapping
        label_map = {name: i for i, name in enumerate(label_encoder.classes_)}
        with open(LABEL_MAP_FILE, 'w') as f:
            json.dump(label_map, f, indent=4)
        
        # 4. Latih model
        model, history = train_keras_model(face_images, labels_int, num_classes, MODEL_H5_FILE)
        
        # 5. Fine-tuning (opsional)
        X_train, X_val, y_train, y_val = train_test_split(
            face_images, labels_int, test_size=0.2, random_state=42, stratify=labels_int)
        model, fine_tune_history = fine_tune_model(model, model.layers[1], X_train, y_train, X_val, y_val)
        
        # 6. Evaluasi
        plot_training_history(history)
        plot_confusion_matrix(model, X_val, y_val, label_encoder.classes_)
        
        # 7. Simpan model final
        model.save(MODEL_H5_FILE)

    except Exception as e:
        print(f"Error: {e}")
