# Drowsiness Detector

Sistem deteksi ngantuk pengemudi secara *real time* yang dibangun menggunakan **Python**, **OpenCV**, dan **Dlib 68-Landmarks Eye Aspect Ratio (EAR)**. Project ini secara otomatis mengunduh file model yang diperlukan dan memanfaatkan akselerasi hardware (Apple MPS / NVIDIA CUDA / CPU) secara dinamis.

---

## 🌟 Fitur Utama

- **68 Facial Landmark EAR Calculation**: Menghitung **Eye Aspect Ratio (EAR)** secara presisi untuk memantau tingkat keterbukaan mata dan kedipan secara akurat.
- **Auto Model Downloader**: Mengunduh dan memverifikasi model Dlib `shape_predictor_68_face_landmarks.dat` secara otomatis saat pertama kali dijalankan.
- **Dynamic Eye Contour Visualization**: Menandai titik-titik wajah di area mata dengan indikator warna dinamis:
  - 🟢 **Green**: Bangun / Terjaga (`MELEK`)
  - 🟠 **Orange**: Peringatan Mengantuk (`TERDETEKSI NGANTUK`)
  - 🔴 **Red**: Peringatan Tertidur (`TERDETEKSI TIDUR`)
- **Cross-Platform Accelerator Detection**: Memilih secara otomatis antara GPU Apple Silicon (`MPS`), GPU NVIDIA (`CUDA`), or `CPU`.

---

## 🛠️ Tech Stack

- **Python 3.10+ / 3.11+ / 3.14+**
- **OpenCV (`opencv-python`)**
- **Dlib**
- **SciPy**
- **PyTorch (`torch`)**

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/drowsiness-detector.git
cd drowsiness-detector
```

### 2. Buat dan Aktifkan Virtual Environment
```bash
# Create Virtual Environment
python3 -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install opencv-python dlib scipy torch requests
```

---

## 💻 Running the Application

Execute the main script:
```bash
python -m main
```

> **Catatan:** Saat dijalankan pertama kali, script akan secara otomatis mengunduh `shape_predictor_68_face_landmarks.dat` (~99MB) jika file tersebut belum ada di folder project.

Untuk menutup aplikasi, tekan `q`.