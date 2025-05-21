# Laporan Proyek Machine Learning - Muhammad Mahathir

## Project Overview

Sistem rekomendasi telah menjadi komponen penting dalam platform digital modern, membantu pengguna menemukan konten yang relevan di tengah banjirnya informasi. Di industri hiburan khususnya, sistem rekomendasi film menjadi krusial bagi platform streaming dan database film untuk meningkatkan engagement pengguna, retensi pelanggan, dan pendapatan [1].

Permasalahan utama yang dihadapi pengguna platform film adalah "paradoks pilihan" - ketika pengguna dihadapkan dengan ribuan pilihan film, mereka cenderung kesulitan memutuskan apa yang akan ditonton, yang akhirnya dapat menyebabkan ketidakpuasan dan pengalaman negatif [2]. Menurut penelitian dari Nielsen, 70% penonton menghabiskan setidaknya 2-3 menit untuk mencari konten yang akan ditonton, dan 21% penonton menyerah jika tidak menemukan konten yang menarik [3].

Sistem rekomendasi film bekerja untuk mengatasi masalah ini dengan menganalisis preferensi pengguna dan memberikan saran film yang mungkin diminati. Pendekatan yang populer dalam pengembangan sistem rekomendasi film meliputi Content-Based Filtering yang berfokus pada atribut film (seperti genre, aktor, sutradara) dan Collaborative Filtering yang memanfaatkan pola rating dari banyak pengguna [4].

Dalam proyek ini, saya mengembangkan sistem rekomendasi film menggunakan dataset MovieLens Small Latest yang berisi 100.836 rating dari 610 pengguna terhadap 9.742 film. Proyek ini bertujuan untuk mengimplementasikan dan membandingkan dua pendekatan populer dalam sistem rekomendasi – Content-Based Filtering dan Collaborative Filtering – untuk memberikan rekomendasi film yang akurat dan relevan kepada pengguna.

**Referensi:**
1. Smith, B., & Linden, G. (2017). Two Decades of Recommender Systems at Amazon.com. IEEE Internet Computing, 21(3), 12-18.
2. Schwartz, B. (2004). The Paradox of Choice: Why More Is Less. New York: Harper Perennial.
3. Nielsen. (2019). The Nielsen Total Audience Report: Q1 2019. The Nielsen Company.
4. Adomavicius, G., & Tuzhilin, A. (2005). Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734-749.

## Business Understanding

### Problem Statements

Dalam pengembangan sistem rekomendasi film, beberapa permasalahan utama yang perlu diselesaikan:

- Bagaimana cara membantu pengguna menemukan film yang sesuai dengan preferensi mereka di antara ribuan pilihan film yang tersedia?
- Bagaimana mengatasi "cold start problem" ketika belum ada data rating dari pengguna baru atau belum ada rating untuk film baru?
- Bagaimana mengembangkan sistem rekomendasi yang dapat memberikan rekomendasi yang sesuai dengan preferensi unik setiap pengguna?

### Goals

Berdasarkan permasalahan di atas, tujuan dari proyek ini adalah:

- Mengembangkan sistem rekomendasi film yang dapat memberikan rekomendasi film yang relevan dan menarik bagi pengguna berdasarkan preferensi mereka.
- Mengimplementasikan pendekatan Content-Based Filtering untuk mengatasi "cold start problem" dengan merekomendasikan film berdasarkan kemiripan fitur.
- Mengimplementasikan pendekatan Collaborative Filtering untuk memberikan rekomendasi yang dipersonalisasi berdasarkan pola rating pengguna lain.

### Solution Statements

Untuk mencapai tujuan di atas, saya mengajukan dua pendekatan solusi:

1. **Content-Based Filtering**: Mengembangkan sistem rekomendasi yang merekomendasikan film berdasarkan kemiripan fitur film (dalam hal ini genre) dengan film yang disukai pengguna sebelumnya. Pendekatan ini menggunakan TF-IDF Vectorizer untuk mengubah data tekstual genre menjadi representasi vektor numerik, kemudian menggunakan Cosine Similarity untuk menghitung kemiripan antar film.

2. **Collaborative Filtering**: Mengembangkan sistem rekomendasi yang merekomendasikan film berdasarkan preferensi pengguna lain yang memiliki pola rating serupa. Pendekatan ini menggunakan teknik Singular Value Decomposition (SVD) untuk melakukan dimensionality reduction pada matriks user-item dan mengungkap pola tersembunyi dalam data rating.

## Data Understanding

Dataset MovieLens Small Latest adalah dataset yang berisi rating film dari pengguna MovieLens. Dataset ini dikeluarkan oleh GroupLens Research dan merupakan versi kecil dari dataset MovieLens yang lebih besar. Dataset ini dapat diunduh dari https://grouplens.org/datasets/movielens/latest/ atau https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset

Dataset ini terdiri dari beberapa file dengan jumlah data dan kondisi sebagai berikut:

### 1. Dataset Ratings (ratings.csv)

**Jumlah Data:**
- Jumlah baris: 100.836
- Jumlah kolom: 4

**Kondisi Data:**
- Tidak terdapat missing values
- Tidak terdapat data duplikat
- Rating bervariasi dari 0.5 hingga 5.0 dengan interval 0.5
- Rata-rata rating adalah sekitar 3.5

**Uraian Fitur:**
- `userId`: ID unik untuk setiap pengguna (1 sampai 610)
- `movieId`: ID unik untuk setiap film
- `rating`: Rating yang diberikan oleh pengguna (skala 0.5 sampai 5.0 dengan interval 0.5)
- `timestamp`: Waktu rating diberikan (dalam format Unix timestamp)

### 2. Dataset Movies (movies.csv)

**Jumlah Data:**
- Jumlah baris: 9.742
- Jumlah kolom: 3

**Kondisi Data:**
- Tidak terdapat missing values
- Tidak terdapat data duplikat
- Tahun rilis film berkisar dari 1902 hingga 2018

**Uraian Fitur:**
- `movieId`: ID unik untuk setiap film
- `title`: Judul film yang disertai dengan tahun rilis dalam tanda kurung
- `genres`: Genre film yang dipisahkan dengan karakter '|'

### 3. Dataset Tags (tags.csv)

**Jumlah Data:**
- Jumlah baris: 3.683
- Jumlah kolom: 4

**Kondisi Data:**
- Tidak terdapat missing values
- Tidak terdapat data duplikat

**Uraian Fitur:**
- `userId`: ID unik untuk setiap pengguna
- `movieId`: ID unik untuk setiap film
- `tag`: Tag yang diberikan oleh pengguna pada film tertentu
- `timestamp`: Waktu tag diberikan (dalam format Unix timestamp)

### 4. Dataset Links (links.csv)

**Jumlah Data:**
- Jumlah baris: 9.742
- Jumlah kolom: 3

**Kondisi Data:**
- Tidak terdapat missing values
- Tidak terdapat data duplikat

**Uraian Fitur:**
- `movieId`: ID unik untuk setiap film (sama dengan movieId di dataset movies)
- `imdbId`: ID film di database IMDb
- `tmdbId`: ID film di database The Movie Database (TMDb)

### Exploratory Data Analysis (EDA)

Berdasarkan eksplorasi data yang dilakukan, beberapa insight penting dari dataset ini adalah:

#### 1. Analisis Rating Film

Rating bervariasi dari 0.5 hingga 5.0 dengan interval 0.5. Rata-rata rating adalah sekitar 3.5, menunjukkan bahwa pengguna cenderung memberikan rating positif. Rating 4.0 adalah yang paling umum diberikan, diikuti oleh rating 3.0 dan 5.0.

Dari analisis distribusi rating, terlihat bahwa:
- Rating 4.0 adalah yang paling sering diberikan (sekitar 27,000 kali)
- Rating 3.0 dan 5.0 juga cukup populer (masing-masing sekitar 21,000 dan 14,000 kali)
- Rating yang lebih rendah (0.5, 1.0, 1.5) jarang diberikan

Hal ini menunjukkan bahwa pengguna cenderung memberikan rating positif pada film yang mereka tonton, yang mungkin disebabkan oleh bias seleksi (pengguna cenderung menonton film yang mereka pikir akan mereka sukai).

#### 2. Analisis Tahun Rilis Film

Distribusi film berdasarkan tahun rilis menunjukkan peningkatan jumlah film dari tahun ke tahun, dengan jumlah film tertinggi dirilis pada periode 2000-2010.

Dari analisis tahun rilis film, terlihat bahwa:
- Terdapat peningkatan signifikan jumlah film dalam dataset mulai tahun 1990-an
- Puncak jumlah film terjadi sekitar tahun 2000-2010
- Terdapat sedikit penurunan jumlah film setelah tahun 2010, yang mungkin disebabkan oleh keterlambatan dalam penambahan film baru ke database

#### 3. Analisis Genre Film

Genre yang paling populer adalah Drama, diikuti oleh Comedy dan Thriller. Banyak film memiliki lebih dari satu genre.

Dari analisis genre film, terlihat bahwa:
- Drama adalah genre paling dominan dengan lebih dari 4,000 film
- Comedy berada di posisi kedua dengan sekitar 3,500 film
- Thriller, Romance, dan Action juga merupakan genre populer
- Genre seperti Film-Noir, IMAX, dan Documentary relatif lebih jarang

Hal ini memberikan gambaran tentang distribusi genre dalam dataset dan dapat membantu dalam memahami preferensi pengguna secara umum.

#### 4. Analisis Aktivitas Pengguna

Jumlah rating yang diberikan per pengguna bervariasi, dengan rata-rata sekitar 165 rating per pengguna. Namun distribusinya cenderung skewed, dengan beberapa pengguna yang sangat aktif memberikan rating.

Dari analisis aktivitas pengguna, terlihat bahwa:
- Mayoritas pengguna memberikan antara 20-200 rating
- Terdapat beberapa pengguna yang sangat aktif yang memberikan lebih dari 500 rating
- Distribusi jumlah rating per pengguna cenderung right-skewed, menunjukkan bahwa sebagian besar pengguna memberikan rating dalam jumlah moderat, sementara sedikit pengguna memberikan rating dalam jumlah besar

#### 5. Analisis Popularitas Film

Beberapa film menerima rating jauh lebih banyak dibandingkan film lainnya. Film-film populer seperti "Forrest Gump", "Pulp Fiction", dan "The Shawshank Redemption" memiliki jumlah rating tertinggi.

Dari analisis popularitas film, terlihat bahwa:
- Film "Forrest Gump (1994)" adalah yang paling banyak diberi rating, diikuti oleh "The Shawshank Redemption (1994)" dan "Pulp Fiction (1994)"
- Film-film klasik dan blockbuster mendominasi daftar film dengan rating terbanyak
- Terdapat korelasi antara popularitas film secara umum dengan jumlah rating yang diterima dalam dataset

## Data Preparation

Beberapa teknik data preparation yang diterapkan dalam proyek ini:

### 1. Mengkonversi Timestamp

Timestamp dalam format Unix dikonversi menjadi format datetime untuk memudahkan analisis berdasarkan waktu. Konversi ini diperlukan untuk memudahkan analisis temporal, seperti melihat tren rating berdasarkan waktu atau menganalisis pola pengguna dalam memberikan rating pada periode tertentu.

### 2. Penggabungan Dataset

Dataset ratings dan movies digabungkan berdasarkan kolom movieId untuk mendapatkan informasi lengkap tentang film dan rating-nya. Penggabungan ini diperlukan untuk mengakses informasi film (seperti judul dan genre) bersama dengan data rating dalam satu dataset, yang memudahkan analisis dan pengembangan model.

### 3. Pembuatan Matriks User-Item untuk Collaborative Filtering

Untuk model Collaborative Filtering, dibuat matriks user-item (pivot table) yang berisi rating pengguna untuk setiap film. Matriks ini memiliki dimensi (jumlah pengguna x jumlah film), dengan nilai 0 untuk film yang belum diberi rating oleh pengguna. Pembuatan matriks ini merupakan langkah penting dalam persiapan data untuk algoritma collaborative filtering.

### 4. Normalisasi Rating untuk Collaborative Filtering

Untuk model Collaborative Filtering, rating pengguna dinormalisasi dengan mengurangkan nilai rata-rata rating pengguna tersebut. Normalisasi ini penting untuk mengatasi bias pengguna, di mana beberapa pengguna cenderung memberikan rating lebih tinggi atau lebih rendah secara konsisten. Dengan normalisasi, kita mendapatkan nilai rating yang lebih objektif relatif terhadap perilaku rating pengguna tersebut.

### 5. Transformasi Data untuk Content-Based Filtering

Untuk model Content-Based Filtering, fitur genre film ditransformasi menjadi representasi vektor menggunakan TF-IDF Vectorizer. Transformasi ini diperlukan karena algoritma machine learning bekerja dengan data numerik, sementara genre film adalah data tekstual. TF-IDF mengubah data kategorik menjadi representasi numerik yang menangkap makna semantik dari genre film.

### 6. Perhitungan Similarity untuk Content-Based Filtering

Cosine similarity dihitung antara semua pasangan film berdasarkan representasi TF-IDF dari genre mereka. Perhitungan similarity ini diperlukan untuk mengukur kemiripan antar film berdasarkan genre mereka, yang akan digunakan dalam model Content-Based Filtering untuk menemukan film yang mirip.

### 7. Pembuatan Reverse Mapping

Untuk mempermudah pencarian film, dibuat mapping dari judul film ke indeksnya dalam dataframe. Mapping ini memungkinkan kita untuk dengan cepat menemukan indeks film berdasarkan judulnya, yang diperlukan dalam fungsi rekomendasi.

## Modeling

Dalam proyek ini, saya mengimplementasikan dua pendekatan sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering.

### 1. Content-Based Filtering

Content-Based Filtering merekomendasikan item berdasarkan kemiripan fitur dengan item yang disukai pengguna sebelumnya. Dalam konteks film, fitur yang digunakan adalah genre film.

#### Implementasi Model

1. **TF-IDF Vectorization**: Mengubah data tekstual genre menjadi representasi vektor numerik.
2. **Cosine Similarity**: Menghitung kemiripan antar film berdasarkan representasi vektor mereka.
3. **Fungsi Rekomendasi**: Mengambil judul film sebagai input dan mengembalikan film-film yang memiliki kemiripan tertinggi.

#### Hasil Rekomendasi

Berikut contoh rekomendasi untuk film "Toy Story (1995)":

| title | genres | similarity_score |
|-------|--------|-----------------|
| Antz (1998) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Toy Story 2 (1999) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Adventures of Rocky and Bullwinkle, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Emperor's New Groove, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Monsters, Inc. (2001) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Wild, The (2006) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Shrek the Third (2007) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Tale of Despereaux, The (2008) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Asterix and the Vikings (Astérix et les Vikings...) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Turbo (2013) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |

Dari hasil rekomendasi, terlihat bahwa model berhasil merekomendasikan film-film dengan genre yang sama dengan "Toy Story (1995)", yaitu film-film animasi dengan genre Adventure, Animation, Children, Comedy, dan Fantasy.

### 2. Collaborative Filtering

Collaborative Filtering merekomendasikan item berdasarkan preferensi pengguna lain yang memiliki pola rating serupa. Pendekatan ini menggunakan teknik Singular Value Decomposition (SVD) untuk melakukan dimensionality reduction pada matriks user-item.

#### Implementasi Model

1. **Pembuatan Matriks User-Item**: Membuat matriks yang berisi rating pengguna untuk setiap film.
2. **Normalisasi Rating**: Mengurangkan rating dengan rata-rata rating pengguna untuk mengatasi bias.
3. **Singular Value Decomposition (SVD)**: Menerapkan SVD pada matriks yang telah dinormalisasi untuk mengungkap pola tersembunyi dalam data rating.
4. **Prediksi Rating**: Mengalikan kembali matriks hasil SVD untuk mendapatkan prediksi rating untuk semua pasangan pengguna-film.
5. **Fungsi Rekomendasi**: Mengambil ID pengguna sebagai input dan mengembalikan film-film yang belum ditonton dengan prediksi rating tertinggi.

#### Hasil Rekomendasi

Berikut contoh rekomendasi untuk pengguna dengan ID 1:

| title | genres | predicted_rating |
|-------|--------|-----------------|
| Godfather, The (1972) | Crime\|Drama | 5.23 |
| Shawshank Redemption, The (1994) | Crime\|Drama | 5.19 |
| Usual Suspects, The (1995) | Crime\|Mystery\|Thriller | 5.18 |
| Rear Window (1954) | Mystery\|Thriller | 5.17 |
| Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | Comedy\|War | 5.16 |
| Third Man, The (1949) | Film-Noir\|Mystery\|Thriller | 5.15 |
| Paths of Glory (1957) | Drama\|War | 5.14 |
| Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) | Drama\|Film-Noir | 5.14 |
| Maltese Falcon, The (1941) | Film-Noir\|Mystery | 5.13 |
| Double Indemnity (1944) | Crime\|Drama\|Film-Noir | 5.13 |

Dari hasil rekomendasi, terlihat bahwa model berhasil merekomendasikan film-film klasik dengan rating tinggi yang belum ditonton oleh pengguna dengan ID 1. Rekomendasi ini didasarkan pada pola rating pengguna lain yang memiliki preferensi serupa dengan pengguna tersebut.

## Evaluation

### 1. Evaluasi Content-Based Filtering

Untuk mengevaluasi model Content-Based Filtering, saya menggunakan metrik Precision@10, yang mengukur proporsi item yang relevan di antara 10 rekomendasi teratas. Dalam konteks ini, "relevan" didefinisikan sebagai film yang memiliki setidaknya satu genre yang sama dengan film referensi.

Hasil evaluasi menunjukkan bahwa model mencapai Precision@10 rata-rata sebesar 1.0, yang berarti 100% dari rekomendasi memiliki setidaknya satu genre yang sama dengan film referensi. Ini merupakan nilai yang sangat baik, menunjukkan bahwa model dapat merekomendasikan film dengan genre yang relevan secara konsisten.

Namun, perlu dicatat bahwa definisi "relevan" yang digunakan di sini cukup longgar (hanya memerlukan satu genre yang sama). Dalam praktiknya, kita mungkin ingin menggunakan definisi yang lebih ketat, seperti mengharuskan lebih banyak genre yang cocok atau menggunakan metrik lain seperti Jaccard similarity untuk mengukur kemiripan genre.

### 2. Evaluasi Collaborative Filtering

Untuk mengevaluasi model Collaborative Filtering, saya menggunakan metrik Root Mean Squared Error (RMSE), yang mengukur perbedaan antara rating yang diprediksi dan rating yang sebenarnya. RMSE yang lebih rendah menunjukkan prediksi yang lebih akurat.

Hasil evaluasi menunjukkan bahwa model mencapai RMSE sebesar 0.89, yang berarti rata-rata prediksi rating berbeda sekitar 0.89 poin dari rating sebenarnya. Mengingat skala rating adalah 0.5 hingga 5.0, RMSE sebesar 0.89 menunjukkan performa yang cukup baik.

Untuk meningkatkan performa model, beberapa pendekatan yang dapat dicoba:
- Menggunakan jumlah faktor laten yang berbeda dalam SVD
- Menerapkan teknik regularisasi untuk mengatasi overfitting
- Menggunakan algoritma collaborative filtering lain seperti Matrix Factorization atau Neural Collaborative Filtering

## Conclusion

Dalam proyek ini, saya telah berhasil mengimplementasikan dua pendekatan sistem rekomendasi film: Content-Based Filtering dan Collaborative Filtering. Kedua pendekatan ini memiliki kelebihan dan kekurangan masing-masing, dan dapat digunakan dalam skenario yang berbeda.

Content-Based Filtering sangat efektif dalam merekomendasikan film dengan karakteristik serupa berdasarkan fitur film (dalam hal ini genre). Pendekatan ini cocok untuk mengatasi "cold start problem" ketika belum ada data rating dari pengguna baru, karena rekomendasi didasarkan pada kemiripan item, bukan pada pola rating pengguna. Namun, pendekatan ini terbatas pada fitur yang digunakan (dalam hal ini hanya genre) dan tidak dapat menangkap preferensi pengguna yang lebih kompleks.

Collaborative Filtering, di sisi lain, dapat menangkap pola tersembunyi dalam data rating dan memberikan rekomendasi yang lebih dipersonalisasi berdasarkan preferensi pengguna lain yang serupa. Pendekatan ini dapat merekomendasikan film yang mungkin tidak memiliki kemiripan fitur yang jelas dengan film yang disukai pengguna sebelumnya, tetapi disukai oleh pengguna lain dengan preferensi serupa. Namun, pendekatan ini memerlukan data rating yang cukup dan tidak dapat memberikan rekomendasi yang baik untuk pengguna baru atau film baru.

Untuk pengembangan lebih lanjut, pendekatan hybrid yang menggabungkan Content-Based Filtering dan Collaborative Filtering dapat dipertimbangkan untuk memanfaatkan kelebihan kedua pendekatan dan mengatasi keterbatasan masing-masing. Selain itu, fitur tambahan seperti aktor, sutradara, dan sinopsis film dapat digunakan untuk meningkatkan performa Content-Based Filtering, sementara teknik deep learning dapat digunakan untuk meningkatkan performa Collaborative Filtering.
