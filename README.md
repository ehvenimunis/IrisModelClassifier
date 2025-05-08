# Iris Model Classifier

Bu proje, Iris veri seti üzerinde farklı yapay sinir ağı mimarileri ve hiperparametre kombinasyonlarını test eden bir sınıflandırma modelidir.

## 🎯 Proje Özellikleri

### Model Mimarisi
- İki gizli katmanlı yapay sinir ağı
- İki farklı nöron konfigürasyonu:
  - 8-8 nöronlu yapı
  - 16-16 nöronlu yapı
- Test edilen aktivasyon fonksiyonları:
  - Sigmoid
  - ReLU
  - Tanh
  - LeakyReLU
- Çıkış katmanı: 3 nöronlu Softmax aktivasyonu

### Eğitim Özellikleri
- Öğrenme oranları: 0.01, 0.001, 0.0001
- Batch size: 8
- Dropout oranı: 0.3
- Early Stopping (patience=10)
- Veri bölünmesi: %80 eğitim, %20 test
- Özellik normalizasyonu: StandardScaler

### Performans Metrikleri
- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1 Skoru

## 🚀 Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Projeyi çalıştırın:
```bash
python main.py
```

## 📊 Çıktılar

Program her model konfigürasyonu için:
1. Performans metriklerini tablo halinde gösterir
2. Eğitim ve doğrulama sürecindeki:
   - Kayıp (loss) grafiği
   - Doğruluk (accuracy) grafiği
   gösterilir

## 📋 Gereksinimler

- Python 3.7+
- NumPy >= 1.19.2
- Pandas >= 1.2.0
- Matplotlib >= 3.3.2
- Scikit-learn >= 0.24.0
- TensorFlow >= 2.4.0

## 🔍 Kod Yapısı

- `main.py`: Ana program dosyası
  - Model oluşturma ve eğitim fonksiyonları
  - Performans değerlendirme
  - Görselleştirme araçları
- `requirements.txt`: Proje bağımlılıkları

## 📈 Model Değerlendirme

Her model konfigürasyonu için:
1. Farklı öğrenme oranları test edilir
2. Her kombinasyon için performans metrikleri hesaplanır
3. Eğitim süreci görselleştirilir
4. En iyi performans gösteren model konfigürasyonu belirlenir
