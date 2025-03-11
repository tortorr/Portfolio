# Background
Banyak orang di screening PTM, tapi belum tentu menerima screening lengkap sehingga deteksi dini beberapa penyakit bisa terlewat. Akibatnya, bisa aja ada orang yg high risk, namun tidak tertandai sebagai demikian. Oleh karena itu kita pengen explore metode ML untuk risk scoring hipertensi.

# Objective
Mencari variable PTM yang berkorelasi dengan hipertensi.

# Exploratory Data Analysis (EDA)
Sumber data yang digunakan untuk melakukan EDA adalah table gold individu yang sudah dibuat sebelumnya dan digunakan sebagai dasar dalam pembuatan beberapa dashboard PTM selama ini.

# Pre-processing Data
**Langkah pertama** yang dilakukan adalah, memeriksa kelengkapan data individu pada setiap kolom pada tabel `trial-pusdatin-kemenkes.gold_dashboard_ptm.detail_ptm_deteksi_dini_visit` dengan menggunakan bigquery. Setelah itu menyimpan hasil query kedalam table baru `trial-pusdatin-kemenkes.test.ptm_ml`.

**Langkah kedua** melakukan perapihan dan perhitungan terhadap kolom yang mengandung null, kolom yang memiliki persentase nulls lebih dari dan atau 30% diberikan tanda merah, sedangkan kolom yang memiliki persentase nulls kurang dari 30% diberikan tanda hijau.

Untuk kolom yang memiliki persentase nulls lebih dari 30% akan didrop, dari total 77 kolom menjadi 34 kolom.
| Column Name                              | Nulls Count  | Number of Rows | % of Nulls |
|------------------------------------------|-------------|---------------|------------|
| left_ear_presbycusis_moved               | 82,565,207  | 82,566,819    | 100.00%    |
| right_eye_retinopathy_moved              | 82,565,205  | 82,566,819    | 100.00%    |
| right_ear_presbycusis_moved              | 82,565,204  | 82,566,819    | 100.00%    |
| right_eye_glaucoma_moved                 | 82,565,200  | 82,566,819    | 100.00%    |
| left_ear_cerumen_moved                   | 82,565,191  | 82,566,819    | 100.00%    |
| right_ear_cerumen_moved                  | 82,565,189  | 82,566,819    | 100.00%    |
| left_eye_glaucoma_moved                  | 82,565,184  | 82,566,819    | 100.00%    |
| left_eye_retinopathy_moved               | 82,565,184  | 82,566,819    | 100.00%    |
| hba1c                                    | 82,470,552  | 82,566,819    | 99.88%     |
| ldl                                      | 82,452,096  | 82,566,819    | 99.86%     |
| hdl                                      | 82,443,660  | 82,566,819    | 99.85%     |
| triglyceride                             | 82,439,627  | 82,566,819    | 99.85%     |
| gula_darah_pp                            | 82,382,193  | 82,566,819    | 99.78%     |
| gula_darah_puasa                         | 81,944,774  | 82,566,819    | 99.25%     |
| diagnosis_3                              | 81,923,563  | 82,566,819    | 99.22%     |
| uric_acid                                | 80,528,351  | 82,566,819    | 97.53%     |
| cholesterol_total                        | 80,451,838  | 82,566,819    | 97.44%     |
| diagnosis_2                              | 80,140,236  | 82,566,819    | 97.06%     |
| sadanis_right_check                      | 79,870,180  | 82,566,819    | 96.73%     |
| iva_check                                | 77,930,435  | 82,566,819    | 94.38%     |
| sadanis_left_check                       | 71,589,762  | 82,566,819    | 86.71%     |
| patient_village                          | 68,951,151  | 82,566,819    | 83.51%     |
| smoke_exposure                           | 68,633,216  | 82,566,819    | 83.12%     |
| diagnosis_1                              | 67,545,234  | 82,566,819    | 81.81%     |
| telephone_number                         | 64,196,233  | 82,566,819    | 77.75%     |
| nakes_id                                 | 64,187,235  | 82,566,819    | 77.74%     |
| nakes_role                               | 63,904,358  | 82,566,819    | 77.40%     |
| is_referred_to_secondary_healthcare      | 57,840,200  | 82,566,819    | 70.05%     |
| right_ear_pus_moved                      | 55,095,530  | 82,566,819    | 66.73%     |
| left_ear_pus_moved                       | 54,922,949  | 82,566,819    | 66.52%     |
| left_ear_ha_moved                        | 54,889,681  | 82,566,819    | 66.48%     |
| left_eye_cataract_moved                  | 54,889,363  | 82,566,819    | 66.48%     |
| patient_address                          | 50,558,753  | 82,566,819    | 61.23%     |
| gula_darah_sewaktu                       | 46,077,690  | 82,566,819    | 55.81%     |
| marriage_status                          | 30,889,582  | 82,566,819    | 37.41%     |
| patient_job                              | 24,424,364  | 82,566,819    | 29.58%     |
| belly_circumference_cm                   | 21,178,673  | 82,566,819    | 25.65%     |
| puskesmas_district                       | 14,452,458  | 82,566,819    | 17.50%     |
| height_cm                                | 14,017,632  | 82,566,819    | 16.98%     |
| weight_kg                                | 13,911,942  | 82,566,819    | 16.85%     |
| age_year                                 | 179,150     | 82,566,819    | 0.22%      |
| citizen_id                               | 125,358     | 82,566,819    | 0.15%      |
| patient_name                             | 96,100      | 82,566,819    | 0.12%      |
| birth_date                               | 79,738      | 82,566,819    | 0.10%      |
| patient_gender                           | 79,702      | 82,566,819    | 0.10%      |
| masked_citizen_id                        | 79,702      | 82,566,819    | 0.10%      |
| puskesmas_name                           | 0           | 82,566,819    | 0.00%      |
| checkup_id                               | 0           | 82,566,819    | 0.00%      |
| checkup_date                             | 0           | 82,566,819    | 0.00%      |
| hypertension_flag                        | 0           | 82,566,819    | 0.00%      |
| obesity_flag                             | 0           | 82,566,819    | 0.00%      |
| is_unclean_data                          | 0           | 82,566,819    | 0.00%      |
| database                                 | 0           | 82,566,819    | 0.00%      |

**Langkah ketiga** melakukan pengecekan perhitungan kembali terhadap kolom yang mengandung null, tetapi kali ini data akan difilter positif hipertensi, dan untuk kolom yang memiliki persentase nulls lebih dari dan atau 30% akan diberikan tanda merah dan didrop.

dari total 77 kolom menjadi 38 kolom.
| col_name                          | nulls_count | percent_nulls |
|-----------------------------------|------------|--------------|
| diagnosis_1                       | 3,414,430  | 100.00%      |
| patient_city                      | 3,414,430  | 100.00%      |
| patient_district                  | 3,414,430  | 100.00%      |
| patient_province                  | 3,414,430  | 100.00%      |
| patient_village                   | 3,414,430  | 100.00%      |
| diagnosis_3                       | 3,414,153  | 99.99%       |
| left_eye_cataract_moved           | 3,414,025  | 99.99%       |
| right_eye_retinopathy_moved       | 3,414,025  | 99.99%       |
| right_eye_glaucoma_moved          | 3,414,024  | 99.99%       |
| left_eye_retinopathy_moved        | 3,414,023  | 99.99%       |
| left_ear_presbycusis_moved        | 3,414,022  | 99.99%       |
| right_ear_presbycusis_moved       | 3,414,021  | 99.99%       |
| right_eye_cataract_moved          | 3,414,021  | 99.99%       |
| right_ear_cerumen_moved           | 3,414,020  | 99.99%       |
| left_eye_glaucoma_moved           | 3,414,020  | 99.99%       |
| left_ear_cerumen_moved            | 3,414,019  | 99.99%       |
| left_ear_pus_moved                | 3,414,019  | 99.99%       |
| left_ear_ha_moved                 | 3,414,019  | 99.99%       |
| right_ear_pus_moved               | 3,414,018  | 99.99%       |
| right_ear_ha_moved                | 3,414,013  | 99.99%       |
| diagnosis_2                       | 3,413,295  | 99.97%       |
| hba1c                              | 3,402,026  | 99.64%       |
| ldl                                | 3,397,063  | 99.49%       |
| hdl                                | 3,395,444  | 99.44%       |
| triglyceride                       | 3,394,553  | 99.42%       |
| gula_darah_pp                      | 3,383,499  | 99.09%       |
| is_referred_to_secondary_healthcare | 3,297,567  | 96.58%       |
| gula_darah_puasa                   | 3,261,432  | 95.52%       |
| iva_check                          | 3,249,919  | 95.18%       |
| sadanis_left_check                 | 3,023,017  | 88.54%       |
| sadanis_right_check                | 3,020,139  | 88.45%       |
| uric_acid                          | 2,920,906  | 85.55%       |
| cholesterol_total                   | 2,906,364  | 85.12%       |
| right_eye_va                       | 2,773,394  | 81.23%       |
| left_eye_va                        | 2,753,435  | 80.64%       |
| telephone_number                   | 2,029,446  | 59.44%       |
| left_ear_ha                        | 1,787,067  | 52.34%       |
| right_ear_ha                       | 1,776,570  | 52.03%       |
| gula_darah_sewaktu                 | 1,324,538  | 38.79%       |
| smoke_exposure                     | 863,316    | 25.28%       |
| puma_level                         | 796,311    | 23.32%       |
| smoking_level                      | 635,906    | 18.62%       |
| salt_exposure                      | 580,128    | 16.99%       |
| sugar_exposure                     | 577,761    | 16.92%       |
| cooking_oil_exposure               | 547,508    | 16.04%       |
| healthy_food_habit                 | 436,943    | 12.80%       |
| belly_circumference_cm             | 428,743    | 12.56%       |
| alcohol_exposure                   | 346,640    | 10.15%       |
| physical_activity                   | 344,751    | 10.10%       |
| height_cm                          | 292,529    | 8.57%        |
| weight_kg                          | 280,079    | 8.20%        |
| nakes_id                           | 64,798     | 1.90%        |
| puskesmas_city                     | 47,866     | 1.40%        |
| puskesmas_district                 | 47,866     | 1.40%        |
| puskesmas_province                 | 47,866     | 1.40%        |
| puskesmas_id                       | 47,856     | 1.40%        |
| faskes_id                          | 47,856     | 1.40%        |
| patient_address                    | 44,793     | 1.31%        |
| diastole_mmhg                      | 22,068     | 0.65%        |
| age_year                           | 17,160     | 0.50%        |
| citizen_id                         | 16,927     | 0.50%        |
| patient_job                        | 16,087     | 0.47%        |
| patient_name                       | 16,083     | 0.47%        |
| patient_gender                     | 16,079     | 0.47%        |
| masked_citizen_id                  | 16,079     | 0.47%        |
| marriage_status                    | 16,079     | 0.47%        |
| birth_date                         | 16,079     | 0.47%        |
| systole_mmhg                       | 2,147      | 0.06%        |
| nakes_role                         | 123        | 0.00%        |
| nakes_role_id                      | 123        | 0.00%        |
| puskesmas_name                     | 0          | 0.00%        |
| checkup_id                         | 0          | 0.00%        |
| checkup_date                       | 0          | 0.00%        |
| hypertension_flag                   | 0          | 0.00%        |
| obesity_flag                        | 0          | 0.00%        |
| is_unclean_data                     | 0          | 0.00%        |
| database                            | 0          | 0.00%        |

**Langkah empat** menentukan variabel akhir yang akan digunakan untuk proses machine learning
| Col_name               | Type    | Value       | Deskripsi |
|------------------------|---------|------------|-----------|
| hypertension_flag      | INTEGER | 0 or 1     | Sebagai flagging atau penanda jika seseorang hipertensi atau tidak (1 = hipertensi | 0 = tidak hipertensi) |
| puskesmas_province     | STRING  | 31         | Kode dagri provinsi (2 digit) |
| puskesmas_city        | STRING  | 3173       | Kode dagri kota/kabupaten (4 digit) |
| masked_citizen_id     | STRING  | 01-9977941 | Citizen_id yang sudah dimasking |
| patient_gender        | STRING  | male or female | Menunjukkan jenis kelamin seseorang (male = pria | female = wanita) |
| patient_job           | STRING  | KARYAWAN / PEGAWAI | Menunjukkan pekerjaan seseorang |
| marriage_status       | STRING  | menikah    | Menunjukkan status pernikahan seseorang |
| age_year             | INTEGER | 25 tahun   | Menunjukkan umur seseorang dalam tahun |
| height_cm            | NUMERIC | 170 cm     | Menunjukkan tinggi seseorang dalam cm |
| weight_kg            | NUMERIC | 70 kg      | Menunjukkan berat badan seseorang dalam kg |
| systole_mmhg         | NUMERIC | 120 mmHg   | Menunjukkan tekanan darah pada saat jantung memompa darah ke dalam pembuluh nadi (saat jantung mengkerut) seseorang |
| diastole_mmhg        | NUMERIC | 80 mmHg    | Menunjukkan tekanan darah pada saat jantung mengembang dan menyedot darah kembali atau pembuluh nadi mengempis kosong |
| belly_circumference_cm | NUMERIC | 80 cm    | Menunjukkan lingkar perut seseorang |
| healthy_food_habit   | NUMERIC  | 0,1,2 or 3 | Menunjukkan kebiasaan makan seseorang |
| puma_level          | INTEGER  | 0 to 7     | Menunjukkan kondisi kesehatan paru-paru seseorang |
| smoking_level       | NUMERIC  | 0 to 3     | Menunjukkan kondisi merokok seseorang |
| smoke_exposure      | NUMERIC  | 1 to 3     | Menunjukkan kondisi terpapar asap rokok seseorang |
| salt_exposure       | NUMERIC  | 0 to 3     | Menunjukkan kondisi konsumsi garam seseorang |
| sugar_exposure      | NUMERIC  | 0 to 3     | Menunjukkan kondisi konsumsi gula seseorang |
| cooking_oil_exposure | NUMERIC  | 0 to 3    | Menunjukkan kondisi konsumsi minyak seseorang |
| alcohol_exposure    | NUMERIC  | 0 to 3     | Menunjukkan kondisi konsumsi alkohol seseorang |
| physical_activity   | NUMERIC  | 0 to 3     | Menunjukkan tingkat olahraga fisik seseorang |
| obesity_flag        | INTEGER  | 0 or 1     | Sebagai flagging atau penanda jika seseorang obesitas atau tidak (1 = obesitas | 0 = tidak obesitas) |

# Analysis
## Executive Summary

- Terdapat sekitar 82,9 juta pemeriksaan PTM yang tercatat di ASIK sampai dengan periode 10 juli 2023. 3,4 juta diantaranya atau sekitar 4,1% adalah pengidap hipertensi.
- Sebanyak 49,3 juta atau 59,5% adalah perempuan, dan sebanyak 33,5 juta atau 40,5% adalah laki-laki.

## Profiling Data PTM
Terdapat 82.995.270 pemeriksaan PTM dimana sebanyak 3.414.430  atau 4.12% pemeriksaan menunjukkan positif hipertensi, dan sebanyak 79.580.840 atau 95,9% pemeriksaan menunjukkan negatif hipertensi.
<p align="center">
  <img src="https://github.com/user-attachments/assets/26ca4bdd-7835-4c71-9f5a-6246a764a2ee" width=600>
  <br><em>Figure 1 | Workflow diagram illustrating data processing steps</em>
</p>
Berikut adalah komposisi pemeriksaan PTM berdasarkan jenis kelamin, sebanyak 49.348.608 pemeriksaan atau 59.,46% adalah perempuan dan sebanyak 33.567.098 atau 40,44% adalah Laki-laki.

<br><br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e7224fe-94e9-416f-a6ef-8e41b1b11e82" width=600>
  <br><em>Figure 2 | Barchart Pemeriksaan PTM berdasarkan Jenis Kelamin</em>
</p>
Berikut adalah sebaran pemeriksaan PTM berdasarkan umur, dengan rentang umur antara 0 - 2022, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.

<br><br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8438b03d-fb61-4182-a0bb-de152feb5503" width=400 height=800>
  <br><em>Figure 3 | Boxplot Pemeriksaan PTM berdasarkan Umur</em>
</p>

## Machine Learning
Langkah selanjutnya adalah melakukan cleaning dan penyesuaian terhadap dataset yang akan digunakan untuk mencari korelitas antara hipertensi dengan variabel lainnya dan machine learning, seperti melakukan replace terhadap value-value yang null, resampling populasi terhadap jumlah jenis kelamin, dll.

dataset  = `trial-pusdatin-kemenkes.test.ptm_ml_cleaned`
| Field                   | Data to Change                           |
|-------------------------|-----------------------------------------|
| hypertension_flag       | none                                    |
| checkup_id             | none                                    |
| puskesmas_province     | null or 01 → 'tidak diketahui'          |
| puskesmas_city         | null or 1101 → 'tidak diketahui'        |
| masked_citizen_id      | none                                    |
| patient_gender         | none                                    |
| age_year              | age_year between 15 and 100             |
| height_cm             | height_cm between 120 and 230           |
| weight_kg             | weight_kg between 35 and 200            |
| systole_mmhg          | systole_mmhg between 50 and 250         |
| diastole_mmhg         | diastole_mmhg between 50 and 250        |
| belly_circumference_cm | belly_circumference_cm between 40 and 120 |
| healthy_food_habit    | null → 0                                |
| puma_level           | null → 2                                |
| smoking_level        | null → 0                                |
| smoke_exposure      | null → 1                                |
| salt_exposure       | null → 0                                |
| sugar_exposure      | null → 0                                |
| cooking_oil_exposure | null → 0                                |
| alcohol_exposure    | null → 0                                |
| physical_activity   | null → 0                                |
| obesity_flag        | none                                    |

## Korelitas Hipertensi
Berikut adalah pemeriksaan hipertensi berdasarkan provinsi. Provinsi 33 merupakan provinsi dengan positif hipertensi terbanyak, sebanyak 406.295 (24,42%) positif hipertensi dari 1.663.712 pemeriksaan.

Provinsi 31 merupakan provinsi dengan positif hipertensi terkecil, sebanyak 486 (0,02%) positif hipertensi dari 2.106.294 pemeriksaan.

Provinsi 32 merupakan provinsi dengan pemeriksaan terbanyak, sebanyak 8.417.310 pemeriksaan, sedangkan provinsi 91 merupakan provinsi dengan pemeriksaan terkecil, sebanyak 43.185 pemeriksaan.
<p align="center">
  <img src="https://github.com/user-attachments/assets/c4d19d45-d464-4148-a644-c0a268af12a7" width=600>
  <br><em>Figure 4 | Barchart pemeriksaan Hipertensi berdasarkan Kode Provinsi</em>
</p>

<br>

Berikut adalah komposisi pemeriksaan hipertensi berdasarkan jenis kelamin, sebanyak 1.058.691 (5,12%) positif hipertensi dari total 20.657.465 pemeriksaan terhadap laki-laki, sedangkan sebanyak 1.126.361 (5,45%) positif hipertensi dari total 20.657.465 pemeriksaan terhadap perempuan.
<p align="center">
  <img src="https://github.com/user-attachments/assets/9ddb049a-a0e1-4495-83d6-0415c73df4c6" width=600>
  <br><em>Figure 5 | Barchart pemeriksaan Hipertensi berdasarkan Jenis Kelamin</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan umur,  dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/90c2250d-d8cf-4753-8382-b5f46b791193" width=600>
  <br><em>Figure 6 | Boxplot pemeriksaan Hipertensi berdasarkan Umur</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan tinggi cm,  dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/00f303bb-039e-4478-82c6-452349b19f4b" width=600>
  <br><em>Figure 7 | Boxplot pemeriksaan Hipertensi berdasarkan Tinggi badan dalam Centimeter</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan berat kg, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/1b6051e9-37d5-4a3c-803f-26b6953e89bd" width=600>
  <br><em>Figure 8 | Boxplot pemeriksaan Hipertensi berdasarkan Berat badan dalam Kilogram</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Systole mmHg, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/db753bd3-6398-4629-9322-3297ba8e14b4" width=600>
  <br><em>Figure 9 | Boxplot pemeriksaan Hipertensi berdasarkan Systole</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Diastole mmHg, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/1fb445a0-e636-4213-b92f-694a88e41ec0" width=600>
  <br><em>Figure 10 | Boxplot pemeriksaan Hipertensi berdasarkan Diastole</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Belly Circumfance, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/2cc2c4e3-358f-4b5f-bca8-7b4024b7eb92" width=600>
  <br><em>Figure 11 | Boxplot pemeriksaan Hipertensi berdasarkan Belly Circumference</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Healthy Food Habit, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/91e0c940-9173-4691-8852-e1be65e637ff" width=600>
  <br><em>Figure 12 | Boxplot pemeriksaan Hipertensi berdasarkan Healthy Food Habit</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Puma Level, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/5be89e24-0f26-4c6d-b1b5-79d6f2c7efdf" width=600>
  <br><em>Figure 13 | Boxplot pemeriksaan Hipertensi berdasarkan Puma Level</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Smoking Level, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/ead21302-6c87-40cf-b073-24513e050248" width=600>
  <br><em>Figure 14 | Boxplot pemeriksaan Hipertensi berdasarkan Smoking Level</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Smoke Exposure, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/df12ec8d-ce33-4c16-8bc5-031377757024" width=600>
  <br><em>Figure 15 | Boxplot pemeriksaan Hipertensi berdasarkan Smoking Exposure</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Salt Exposure, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/38c7652d-a829-4122-b09c-70d815810e34" width=600>
  <br><em>Figure 16 | Boxplot pemeriksaan Hipertensi berdasarkan Salt Exposure</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Sugar Exposure, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/bf70e661-2dd0-4d6a-9dca-0129082f5c06" width=600>
  <br><em>Figure 17 | Boxplot pemeriksaan Hipertensi berdasarkan Sugar Exposure</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Cooking Oil Exposure, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/548bb137-2e60-46e5-8d74-afc4946822f3" width=600>
  <br><em>Figure 18 | Boxplot pemeriksaan Hipertensi berdasarkan Cooking Oil Exposure</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Alcohol Exposure, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/347770fc-4220-4474-ba6c-a604c7dbcfc5" width=600>
  <br><em>Figure 19 | Boxplot pemeriksaan Hipertensi berdasarkan Alcohol Exposure</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Physical Activity, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/6af909f6-f883-438a-8459-f59fcd8a3d06" width=600>
  <br><em>Figure 20 | Boxplot pemeriksaan Hipertensi berdasarkan Physical Activity</em>
</p>

<br>

Berikut adalah sebaran pemeriksaan hipertensi berdasarkan Obesity, dengan beberapa keterangan yang dapat dilihat pada grafik dibawah.
<p align="center">
  <img src="https://github.com/user-attachments/assets/bac4dedf-713d-47a2-be61-38b5c51cdab7" width=600>
  <br><em>Figure 21 | Boxplot pemeriksaan Hipertensi berdasarkan Obesity</em>
</p>

# Appendix
dataset yang digunakan:
- `trial-pusdatin-kemenkes.test.ptm_ml`
- `trial-pusdatin-kemenkes.test.ptm_ml_clean`

# Query
Tabel pembentuk kolom untuk analisis PTM <!--trial-pusdatin-kemenkes.gold_dashboard_ptm.detail_ptm_deteksi_dini_visit-->
```sql
select
  puskesmas_id,
  puskesmas_name,
  puskesmas_province,
  puskesmas_city,
  puskesmas_district,
  patient_province,
  patient_city,
  patient_district,
  faskes_id,
  citizen_id,
  masked_citizen_id,
  patient_name,
  patient_job,
  patient_gender,
  birth_date,
  checkup_id,
  checkup_date,
  height_cm,
  weight_kg,
  age_year,
  gula_darah_sewaktu,
  puma_level,
  marriage_status,
  left_eye_va,
  right_eye_va,
  left_ear_ha,
  right_ear_ha,
  belly_circumference_cm,
  sugar_exposure,
  salt_exposure,
  cooking_oil_exposure,
  smoking_level,
  healthy_food_habit,
  physical_activity,
  alcohol_exposure,
  systole_mmhg,
  diastole_mmhg,
  hypertension_flag,
  obesity_flag,
  is_unclean_data,
  database,
from
  `detail_ptm_deteksi_dini_visit`
where
  true
  and checkup_date <= "2023-07-10"
qualify
  row_number()over(partition by checkup_id order by checkup_date desc) = 1
```

Query pengecekan variabel hipertensi <!--detail_ptm_deteksi_dini_visit-->
```sql
with master as(  
  select
    puskesmas_id,
    puskesmas_name,
    puskesmas_province,
    puskesmas_city,
    puskesmas_district,
    patient_province,
    patient_city,
    patient_district,
    faskes_id,
    citizen_id,
    masked_citizen_id,
    patient_name,
    patient_job,
    patient_gender,
    birth_date,
    checkup_id,
    checkup_date,
    height_cm,
    weight_kg,
    age_year,
    belly_circumference_cm,
    sugar_exposure,
    salt_exposure,
    cooking_oil_exposure,
    smoking_level,
    healthy_food_habit,
    physical_activity,
    alcohol_exposure,
    systole_mmhg,
    diastole_mmhg,
    hypertension_flag,
    obesity_flag,
    is_unclean_data,
    database,
  from
    `detail_ptm_deteksi_dini_visit`
  where
    true
    and checkup_date <= "2023-07-10"
    and hypertension_flag = 1
  qualify
    row_number()over(partition by checkup_id order by checkup_date desc) = 1
)
SELECT col_name,
       COUNT(1) AS nulls_count,
       round(100*(count(1)/
                      (SELECT count(*)
                       FROM master)), 2) AS percent_nulls
FROM master t,
     UNNEST(REGEXP_EXTRACT_ALL(TO_JSON_STRING(t), r'"(\w+)":null')) col_name
GROUP BY col_name
ORDER BY nulls_count DESC
```

Query untuk analisis EDA - Profiling <!--detail_ptm_deteksi_dini_visit-->
```sql
select
  puskesmas_id,
  puskesmas_name,
  puskesmas_province,
  puskesmas_city,
  puskesmas_district,
  patient_province,
  patient_city,
  patient_district,
  faskes_id,
  citizen_id,
  masked_citizen_id,
  patient_name,
  patient_job,
  patient_gender,
  birth_date,
  checkup_id,
  checkup_date,
  height_cm,
  weight_kg,
  age_year,
  belly_circumference_cm,
  sugar_exposure,
  salt_exposure,
  cooking_oil_exposure,
  smoking_level,
  healthy_food_habit,
  physical_activity,
  alcohol_exposure,
  systole_mmhg,
  diastole_mmhg,
  hypertension_flag,
  obesity_flag,
  is_unclean_data,
  database,
from
  `detail_ptm_deteksi_dini_visit`
where
  true
  and checkup_date <= "2023-07-10"
qualify
  row_number()over(partition by checkup_id order by checkup_date desc) = 1
```

Query untuk dataset machine learning <!--trial-pusdatin-kemenkes.test.ptm_ml-->
```sql
-- create or replace table test.ptm_ml_cleaned as
with master as( 
  select
    hypertension_flag,
    checkup_id,
    case
      when puskesmas_province is null or puskesmas_province = '01'
      then 'tidak diketahui'
      else puskesmas_province
    end as puskesmas_province,
    case
      when puskesmas_city is null or puskesmas_city = '0101'
      then 'tidak diketahui'
      else puskesmas_city
    end as puskesmas_city,
    masked_citizen_id,
    patient_gender,
    age_year,
    height_cm,
    weight_kg,
    systole_mmhg,
    diastole_mmhg,
    belly_circumference_cm,
    case
      when healthy_food_habit is null
      then 0
      else healthy_food_habit
    end as healthy_food_habit,
    case
      when puma_level is null
      then 2
      else puma_level
    end as puma_level,
    case
      when smoking_level is null
      then 0
      else smoking_level
    end as smoking_level,
    case
      when smoke_exposure is null
      then 1
      else smoke_exposure
    end as smoke_exposure,
    case
      when salt_exposure is null
      then 0
      else salt_exposure
    end as salt_exposure,
    case
      when sugar_exposure is null
      then 0
      else sugar_exposure
    end as sugar_exposure,
    case
      when cooking_oil_exposure is null
      then 0
      else cooking_oil_exposure
    end as cooking_oil_exposure,
    case
      when alcohol_exposure is null
      then 0
      else alcohol_exposure
    end as alcohol_exposure,
    case
      when physical_activity is null
      then 0
      else physical_activity
    end as physical_activity,
    obesity_flag,
  from
    `test.ptm_ml`
  where
    true
    and is_unclean_data = 0
    and age_year between 15 and 100
    and height_cm between 120 and 230
    and weight_kg between 35 and 200
    and systole_mmhg between 50 and 250
    and diastole_mmhg between 50 and 250
    and belly_circumference_cm between 40 and 120
)
,female as(
  select
    *
  from
    master
  where
    true
    and patient_gender = 'female'
  limit
    20657465
)
,male as(
  select
    *
  from
    master
  where
    true
    and patient_gender = 'male'
)
select
  *
from
  female
union all
select
  *
from
  male
```
