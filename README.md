# LLVIP: Düşük Işıkta RGB–IR Multimodal İnsan Tespiti

**İrem Karayel** tarafından Kocaeli Üniversitesi **Derin Öğrenme** dersi kapsamında hazırlanan bu depo, **vize (veri toplama, inceleme, temizleme, ön işleme ve EDA)** çalışmalarını içerir. [LLVIP (Low-Light Visible-Infrared Person)](https://github.com/bupt-ai-cz/LLVIP) veri kümesindeki eşleşmiş görünür ışık (RGB) ve kızılötesi (IR) görüntüler kullanılmıştır.

> **Dikkat:** Bu aşamada model eğitilmemiştir. Precision, Recall, AP/mAP veya çıkarım hızı sonuçları yoktur. Final aşamasında RGB-only, IR-only ve RGB+IR erken füzyon insan tespiti karşılaştırılacaktır.

## 1. Veri özeti ve temel bulgular

| Gösterge | Sonuç |
|---|---:|
| Eşleşmiş RGB–IR görüntü çifti | 15.488 |
| RGB / IR görüntü sayısı | 15.488 / 15.488 |
| Pascal VOC XML dosyası | 15.488 |
| Kaynak görüntü çözünürlüğü | 1280 × 1024 |
| Sınıf | `person` (tek sınıf) |
| Orijinal train / resmî test | 12.025 / 3.463 çift |
| Ham / pozitif alanlı anotasyon kutusu | 42.437 / 42.432 |
| Geçersiz kutu / kutusuz görüntü | 5 / 2 |
| Nihai train / validation / test | 9.937 / 2.088 / 3.463 çift |
| Nihai kutu sayıları (train / val / test) | 28.215 / 5.915 / 8.302 |

- **Eşleştirme:** Altı haneli dosya kimliği (`stem`) temel alınarak RGB, IR ve XML kayıtları bire bir eşleştirilmiştir. Dosya düzeyinde eksik modalite gözlenmemiştir.
- **Anotasyon temizliği:** Sıfır genişlik/yüksekliğe sahip beş kutu işlenmiş tablodan çıkarılmış; ham XML dosyaları ve tüm görüntüler korunmuştur.
- **EDA:** Parlaklık/yoğunluk, kişi sayısı, kutu boyutu ve konumu, RGB–IR ilişkisi, örnek piksel histogramları ve aykırı sahneler incelenmiştir.
- **Ön işleme:** 640×512 küçültme örnekleri ve IR için görüntü bazlı p1–p99 normalizasyonu incelenmiş; normalizasyon varsayılan eğitim tercihi olarak kesinleştirilmemiştir.
- **Veri bölme:** Resmî test korunarak train içerisinden sahne grubu temelli validation oluşturulmuştur. Validation grupları `06`, `11`, `13`, `16` olarak seçilmiştir.
- **Model hazırlığı:** 42.432 kutu YOLO TXT biçimine dönüştürülmüş ve dosyaya yazılmış etiketlerin geri dönüşüm hatası kontrol edilmiştir. Örnek kanal-önce girdiler RGB `(3,512,640)`, IR `(1,512,640)` ve erken füzyon `(4,512,640)` şeklindedir.

**Bilimsel sınır:** Dosya kimliği eşleşmesi kusursuz mekânsal hizalama kanıtı değildir; grup bazlı veri bölmesi tüm fiziksel sahne sızıntısını kesinlikle dışlamaz. Kalite bayrakları hata tanısı değildir. Yerel kontrast ve Laplacian varyansı doğrudan algılanabilirlik/bulanıklık ölçüsü olarak yorumlanmamalıdır.

Nihai bölme dosyası: `data/processed/final_split.csv`

SHA-256:

```text
e893c460f025c3ae5f5fcf9b0a870966d7cd5c4dc17895ebf7ecdaf7600d52d0
```

## 2. Veri kaynağı, erişim ve kullanım koşulları

- **Resmî proje:** https://bupt-ai-cz.github.io/LLVIP/
- **Resmî GitHub/veri indirme bilgileri:** https://github.com/bupt-ai-cz/LLVIP
- **Makale:** Jia ve ark., *LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision*, ICCV Workshops (2021).
- **Lisans ve kullanım şartları:** Resmî depodaki `Term of Use and License` belgesini okuyunuz. Ham görüntü arşivi bu projede yeniden dağıtılmamaktadır.

Bu projedeki yerel dosya envanteri **15.488 çift** içermektedir. İlk makalede farklı sayıda çift bildirildiğinden sürüm karşılaştırması raporda açıklanmıştır. Başka bir veri sürümünde sayımlar aynı olmayabilir.

### Proje Klasör Yapısı

```text
llvip-project/
├── README.md                  # Tek proje kılavuzu
├── requirements.txt           # Doğrudan Python bağımlılıkları
├── requirements.lock          # Geliştirme ortamındaki kesin sürümler
├── .gitignore
├── scripts/                   # Tüm analiz ve veri hazırlığı betikleri (.py)
├── data/
│   ├── raw/
│   │   └── LLVIP_extracted/
│   │       ├── visible/
│   │       │   ├── train/     # <stem>.jpg
│   │       │   └── test/      # <stem>.jpg
│   │       ├── infrared/
│   │       │   ├── train/     # <stem>.jpg
│   │       │   └── test/      # <stem>.jpg
│   │       └── Annotations/  # <stem>.xml
│   └── processed/
│       ├── boxes_cleaned.csv
│       ├── quality_flags.csv
│       ├── final_split.csv
│       ├── step17a_label_manifest.csv
│       └── yolo_labels/{train,val,test}/
└── outputs/
    ├── logs/                  # CSV/JSON sonuçlar
    └── figures/               # PNG görseller
```

`<stem>` altı haneli örnek kimliğidir. Her kimliğe karşılık RGB JPEG, IR JPEG ve bir XML beklenir. Ham görüntüler ve XML'ler değiştirilmeksizin `data/processed/` ve `outputs/` altında çalışma çıktıları üretilir.

**Verinin hazırlanması:** LLVIP veri kümesini resmî bağlantıdan indirip yukarıdaki dizine açın. `scripts/01_download.py`, `gdown` ile indirmeyi dener; erişim/kota nedeniyle manuel indirme gerekebilir. `scripts/02_extract.py`, `data/raw/` içinde **tam bir ZIP dosyası** bekler; dosyalar zaten çıkarılmışsa tekrar çalıştırılmamalıdır. Arşiv açıldıktan sonra `visible`, `infrared`, `Annotations` dizinlerinin doğrudan `LLVIP_extracted` altında bulunduğunu kontrol edin.

## 3. Kurulum (Windows / PowerShell)

Komutları **projenin kök dizininde** çalıştırın. Kullanılan yerel ortam **Python 3.14.2**; `python -m pip check` sonucu **`No broken requirements found.`** olarak bildirilmiştir. Diğer Python sürümlerindeki uyumluluk ayrıca doğrulanmalıdır.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Aynı bağımlılık sürümlerini kullanmak için uyumlu bir ortamda `python -m pip install -r requirements.lock` tercih edilebilir. `requirements.txt` doğrudan kullanılan paketleri, `requirements.lock` ise geliştiricinin çalışan `.venv` ortamındaki **kurulu paketlerin tamamını** kayıt altına alır. Kilit dosyası işletim sistemi/Python sürümüne göre taşınabilir olmayabilir.

Ortamdan kilit dosyası oluşturma:

```powershell
python -m pip freeze > requirements.lock
python -m pip check
python --version
```

**Not:** Eski Windows PowerShell sürümlerindeki yönlendirme bazı dosyaları UTF-16 yazabilir. Kilit dosyası `pip install -r` ile okunamıyorsa UTF-8 olarak yeniden kaydedin. `tkinter` (isteğe bağlı manuel pilot arayüzü) bir pip bağımlılığı değildir.

## 4. Çalıştırma sırası

Aşağıdaki komutlar **kullanılan veri sürümü ve gerekli önceki çıktılar hazırken** geçerlidir. Bazı betikler CSV, JSON veya PNG dosyalarını aynı adlarla yeniden yazar. Mevcut çalışma sonuçlarını korumak için tüm hattı tekrar çalıştırmadan önce yedek alın. Veri arşivini açmadan önce `outputs/logs/` klasörünün varlığını sağlayın.

### 4.1. Temel envanter ve kalite kontrolleri

```powershell
New-Item -ItemType Directory -Force outputs/logs, outputs/figures, data/processed | Out-Null
python scripts/03_inventory.py
python scripts/04_pairing.py
python scripts/05_annotations.py
python scripts/06_quality_pass.py
```

İsteğe bağlı: `scripts/02b_manual_provenance.py` elle indirilmiş verinin kayıtlarını; `scripts/06b_zoom_check.py` seçilmiş görsel yakınlaştırmaları üretir.

### 4.2. Sahne yapısı ve tekrar kayıt analizi

```powershell
python scripts/07_scene_analysis.py
python scripts/08_site_analysis.py
python scripts/09_site_matching.py
python scripts/10_site_assignment.py
python scripts/11_groups_and_duplicates.py
python scripts/12_duplicates_and_calibration.py
```

Bu aşama `data/processed/` altındaki küçük görüntü/benzerlik önbellekleri ve önceki aşamaların kayıtlarına bağlıdır; hesaplama maliyeti yüksek olabilir. Mevcut çıktıları yeniden üretmek gerekmiyorsa atlanabilir.

### 4.3. EDA: keşifsel veri analizi

```powershell
python scripts/eda_01_brightness.py
python scripts/eda_02_person_count.py
python scripts/eda_03_box_dimensions.py
python scripts/eda_04_box_positions.py
python scripts/eda_05_rgb_ir_relationship.py
python scripts/eda_06_paired_pixels.py
python scripts/eda_07_blur_analysis.py
python scripts/eda_08_object_contrast.py
python scripts/eda_08b_metric_audit.py
python scripts/eda_09_contrast_visual_review.py
```

`eda_07` Laplacian varyansı doğrudan bulanıklık tanısı değildir; `eda_08` kontrast metriğinin gerçek kişi görünürlüğü için güvenilir olmadığı örneklerle gösterilmiştir.

### 4.4. Temizleme ve ön işleme

```powershell
python scripts/15a_clean_annotations.py
python scripts/15b_resize_analysis.py
python scripts/15c_ir_normalization.py
python scripts/15d_quality_flags.py
```

`15a` beş geçersiz kutuyu temiz çalışma tablosundan çıkarır. `15b` yalnızca örnek görüntülerde gerçek yeniden boyutlandırma yapar; tüm görüntü dosyalarını küçültmez. `15c` p1–p99 esnetmesinin incelendiği bir deneydir, varsayılan eğitim ayarı değildir.

### 4.5. Sahne grubu temelli veri bölme

```powershell
python scripts/16a_scene_05_22_review.py
python scripts/16b_train_scene_groups.py
python scripts/16c_finalize_validation_groups.py
python scripts/16d_validation_candidates.py
python scripts/16e_create_final_split.py
```

`16e` mevcut sabit split'i değiştirmeye karşı kontroller içerir. `16f_validation_scene_audit.py` ve `16g_targeted_scene_review.py` nihai bölme üzerinde tamamlayıcı kontrollerdir; veri bölmesini değiştirmezler.

### 4.6. YOLO dönüşümü ve doğrulama

```powershell
python scripts/17a_convert_yolo_labels.py
python scripts/17b_verify_yolo_labels.py
python scripts/17c_verify_model_inputs.py
```

`17a` 15.488 TXT etiketi (iki boş TXT dahil) üretir. `17b` 42.432 kutunun kaydedilmiş TXT üzerinden sayısal geri dönüşümünü sınar. `17c` sekiz örnekte model girdi tensörlerini denetler; **model eğitimi yapmaz**.

### 4.7. İsteğe bağlı küçük manuel pilot

`14a_prepare_manual_audit.py`, `14b_select_40.py`, `14b_manual_review.py`, `14c_manual_audit_analysis.py` yalnızca seçilmiş küçük bir örneklemin manuel görsel denetiminde kullanılmıştır. Bu pilot tam otomatik veri hazırlama hattı için zorunlu değildir ve veri setinin genel hata oranını tahmin etmekte kullanılmaz.

## 5. Çıktılar, izlenebilirlik ve teslim

| Yol | İçerik |
|---|---|
| `outputs/logs/` | Envanter, kalite, EDA ve denetim için CSV/JSON özetleri |
| `outputs/figures/` | Gerçek görüntü panelleri ve EDA PNG grafikleri |
| `data/processed/boxes_cleaned.csv` | Geometrik açıdan geçerli nesne kutuları |
| `data/processed/quality_flags.csv` | İnceleme bayrakları; otomatik silme ölçütü değil |
| `data/processed/final_split.csv` | Nihai train/val/test kimlikleri |
| `data/processed/step17a_label_manifest.csv` | RGB, IR ve YOLO etiket yollarını birleştiren manifest |
| `data/processed/yolo_labels/` | Train, val ve test TXT anotasyonları |

Bölmenin hash doğrulaması:

```powershell
(Get-FileHash data\processed\final_split.csv -Algorithm SHA256).Hash.ToLower()
```

### Veri Paylaşımı ve Depo İçeriği

Bu GitHub deposu, LLVIP veri kümesinin incelenmesi, ön işlenmesi ve modellemeye hazırlanması için geliştirilen Python betiklerini, bağımlılık dosyalarını ve çalıştırma yönergelerini içermektedir. Nihai veri bölmesini tanımlayan `data/processed/final_split.csv` dosyası da yeniden üretilebilirliği desteklemek amacıyla paylaşılmaktadır.

Ham LLVIP görüntüleri ve anotasyonları, veri kümesinin kullanım ve dağıtım koşulları nedeniyle depoda yeniden yayımlanmamaktadır. Veri kümesine resmî LLVIP kaynakları üzerinden erişilebilir. Sanal ortam dosyaları, büyük ara çıktılar ve geçici dosyalar da depo kapsamı dışında tutulmuştur.

## 6. Kaynakça

1. X. Jia ve ark., *LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision*, IEEE/CVF ICCV Workshops, 2021.
2. LLVIP resmî veri deposu: https://github.com/bupt-ai-cz/LLVIP
3. Veri seti erişimi: https://bupt-ai-cz.github.io/LLVIP/
4. YOLO nesne tespiti etiket biçimi: https://docs.ultralytics.com/datasets/detect/