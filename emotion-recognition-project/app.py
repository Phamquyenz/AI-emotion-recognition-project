# app.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# --- CÀI ĐẶT BAN ĐẦU ---
MODEL_PATH = os.path.join('saved_models', 'emotion_cnn_model.h5')
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- TẢI MÔ HÌNH VÀ BỘ PHÁT HIỆN KHUÔN MẶT ---
print("Đang tải mô hình...")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Lỗi: không thể tải file mô hình tại '{MODEL_PATH}'.")
    print("Hãy chắc chắn rằng bạn đã chạy script 'src/train_model.py' thành công.")
    exit()

try:
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
except Exception as e:
    print(f"Lỗi: không thể tải file cascade tại '{CASCADE_PATH}'.")
    exit()
print("Tải mô hình thành công!")

def classify_emotion_from_image(image_path):
    """
    Hàm đọc ảnh, phát hiện khuôn mặt, dự đoán cảm xúc và hiển thị kết quả.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("Không tìm thấy khuôn mặt nào trong ảnh.")
    else:
        print(f"Tìm thấy {len(faces)} khuôn mặt.")

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi)
        predicted_emotion = EMOTION_LABELS[prediction.argmax()]
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị ảnh kết quả
    cv2.imshow("Phan loai cam xuc", image)
    print("\nNhấn phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # THAY ĐỔI ĐƯỜNG DẪN NÀY tới ảnh bạn muốn kiểm tra
    input_image_path = 'my_test_image.jpg' 
    
    if not os.path.exists(input_image_path):
        print(f"Lỗi: File ảnh '{input_image_path}' không tồn tại.")
    else:
        classify_emotion_from_image(input_image_path)