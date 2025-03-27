# AI Flow - Platform Pembuatan Model AI

AI Flow adalah platform yang memungkinkan pengguna untuk membuat, melatih, dan mengelola model AI khususnya untuk analisis sentimen melalui API yang mudah digunakan.

## Fitur Utama

- 🤖 Pembuatan model AI otomatis
- 📊 Training model dengan dataset custom
- 📈 Monitoring proses training
- 💾 Download model yang sudah dilatih
- 🔄 API untuk prediksi real-time

## Teknologi yang Digunakan

- FastAPI (Backend Framework)
- PostgreSQL (Database)
- MLflow (Model Tracking)
- Scikit-learn (Machine Learning)
- Joblib (Model Serialization)
- Redis & Celery (Background Tasks)

## Persyaratan Sistem

- Python 3.8+
- PostgreSQL
- Redis
- MLflow

## Instalasi

1. Clone repository:
```bash
git clone <repository_url>
cd ai-flow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup environment variables (.env):
```env
POSTGRES_SERVER=localhost
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=ai_generator
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5001
MODEL_STORAGE_PATH=models
```

4. Jalankan migrasi database:
```bash
alembic upgrade head
```

## Menjalankan Aplikasi

1. Start PostgreSQL:
```bash
brew services start postgresql
```

2. Start Redis:
```bash
brew services start redis
```

3. Start MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5001
```

4. Start FastAPI server:
```bash
uvicorn main:app --reload
```

## Penggunaan API

### 1. Membuat Model Baru

```bash
POST http://localhost:8000/api/v1/models/
Content-Type: multipart/form-data

form-data:
- name: "Sentiment Analysis Model"
- description: "Model untuk analisis sentiment"
- prompt: "Buat model untuk klasifikasi sentiment"
- dataset: [Upload file CSV]
```

### 2. Memeriksa Status Model

```bash
GET http://localhost:8000/api/v1/models/{model_id}/status
```

### 3. Download Model

```bash
GET http://localhost:8000/api/v1/models/{model_id}/download
```

### 4. Prediksi Menggunakan Model

```bash
POST http://localhost:8000/api/v1/models/{model_id}/predict
Content-Type: application/json

{
    "texts": [
        "Pelayanan sangat memuaskan dan ramah",
        "Produk ini mengecewakan dan tidak sesuai ekspektasi"
    ]
}
```

## Format Dataset

Dataset harus dalam format CSV dengan minimal kolom berikut:
- `text`: Teks yang akan dianalisis
- `sentiment`: Label sentiment (positive/negative)

Contoh:
```csv
text,sentiment
Pelayanan sangat memuaskan,positive
Produk ini mengecewakan,negative
```

## Monitoring

1. Cek model yang aktif:
```bash
GET http://localhost:8000/api/v1/models/monitoring/active-trainings
```

2. Cek model yang sudah selesai:
```bash
GET http://localhost:8000/api/v1/models/completed
```

3. Lihat log model:
```bash
GET http://localhost:8000/api/v1/models/{model_id}/logs
```

## Menggunakan Model yang Didownload

```python
import joblib

# Load model dan vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Prediksi
text = "Pelayanan sangat memuaskan"
X = vectorizer.transform([text])
prediction = model.predict(X)[0]
probabilities = model.predict_proba(X)[0]

print(f"Text: {text}")
print(f"Prediction: {prediction}")
print(f"Confidence: {max(probabilities):.2f}")
```

## Struktur Direktori 


## Troubleshooting

1. Database Error:
   - Pastikan PostgreSQL berjalan
   - Periksa kredensial database di .env
   - Jalankan `alembic upgrade head`

2. MLflow Error:
   - Pastikan MLflow server berjalan di port 5001
   - Periksa MLFLOW_TRACKING_URI di .env

3. Model Training Error:
   - Periksa format dataset
   - Pastikan semua kolom yang diperlukan ada
   - Cek log training di endpoint /logs

## Kontribusi

Silakan buat pull request untuk kontribusi atau laporkan issues yang ditemukan.

## Lisensi

[MIT License](LICENSE)