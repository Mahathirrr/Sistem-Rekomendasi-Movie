# Laporan Proyek Machine Learning - Muhammad Mahathir

## Project Overview

Sistem rekomendasi telah menjadi elemen kunci dalam platform digital modern, khususnya di industri hiburan, untuk membantu pengguna menemukan konten yang relevan di tengah banyaknya pilihan. Dalam konteks platform streaming dan database film, sistem rekomendasi film memainkan peran penting dalam meningkatkan *engagement* pengguna, retensi pelanggan, dan pendapatan [1]. Namun, pengguna sering menghadapi "paradoks pilihan," di mana banyaknya opsi film menyebabkan kesulitan dalam memilih tontonan, yang dapat menurunkan kepuasan pengguna [2]. Penelitian dari Nielsen menunjukkan bahwa 70% penonton menghabiskan 2-3 menit untuk mencari konten, dan 21% menyerah jika tidak menemukan film yang menarik [3].

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film yang dapat mengatasi paradoks pilihan dengan memberikan saran film yang relevan berdasarkan preferensi pengguna. Menggunakan dataset **MovieLens Small Latest**, yang berisi 100.836 rating dari 610 pengguna untuk 9.742 film, proyek ini mengimplementasikan dua pendekatan utama: **Content-Based Filtering** (berdasarkan fitur film seperti genre) dan **Collaborative Filtering** (berdasarkan pola rating pengguna). Pendekatan ini dipilih untuk mengeksplorasi efektivitas rekomendasi berdasarkan karakteristik film dan pola perilaku pengguna, sekaligus mengatasi tantangan seperti *cold start problem* untuk pengguna atau film baru.

**Referensi:**
1. B. Smith and G. Linden, "Two Decades of Recommender Systems at Amazon.com," *IEEE Internet Computing*, vol. 21, no. 3, pp. 12-18, 2017.
2. B. Schwartz, *The Paradox of Choice: Why More Is Less*. New York: Harper Perennial, 2004.
3. Nielsen, "The Nielsen Total Audience Report: Q1 2019," The Nielsen Company, 2019.
4. G. Adomavicius and A. Tuzhilin, "Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions," *IEEE Transactions on Knowledge and Data Engineering*, vol. 17, no. 6, pp. 734-749, 2005.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang, permasalahan utama yang dihadapi adalah:
- **Kesulitan memilih film**: Bagaimana membantu pengguna menemukan film yang sesuai dengan preferensi mereka di antara ribuan pilihan film yang tersedia?
- **Cold start problem**: Bagaimana memberikan rekomendasi yang relevan untuk pengguna baru (tanpa riwayat rating) atau film baru (tanpa rating pengguna)?
- **Personalisasi rekomendasi**: Bagaimana mengembangkan sistem yang dapat memberikan rekomendasi yang dipersonalisasi berdasarkan preferensi unik setiap pengguna?

### Goals
Tujuan proyek ini adalah:
- Mengembangkan sistem rekomendasi film yang memberikan saran relevan dan menarik berdasarkan preferensi pengguna untuk meningkatkan pengalaman menonton.
- Mengimplementasikan **Content-Based Filtering** untuk mengatasi *cold start problem* dengan merekomendasikan film berdasarkan kemiripan fitur (genre).
- Mengimplementasikan **Collaborative Filtering** untuk memberikan rekomendasi yang dipersonalisasi berdasarkan pola rating pengguna lain yang serupa.

### Solution Statements
Untuk mencapai tujuan tersebut, dua pendekatan solusi diusulkan:
1. **Content-Based Filtering**: Merekomendasikan film berdasarkan kemiripan fitur genre dengan film yang telah disukai pengguna, menggunakan *TF-IDF Vectorizer* untuk representasi numerik genre dan *Cosine Similarity* untuk mengukur kemiripan antar film.
2. **Collaborative Filtering**: Merekomendasikan film berdasarkan pola rating pengguna lain dengan preferensi serupa, menggunakan *Singular Value Decomposition (SVD)* untuk mengurangi dimensi matriks *user-item* dan mengungkap pola tersembunyi dalam data rating.

## Data Understanding

Dataset **MovieLens Small Latest**, yang dikeluarkan oleh GroupLens Research, berisi 100.836 rating dan 3.683 tag untuk 9.742 film dari 610 pengguna, dikumpulkan antara 29 Maret 1996 hingga 24 September 2018. Dataset ini dirancang untuk tujuan penelitian sistem rekomendasi dan dapat diunduh dari https://grouplens.org/datasets/movielens/latest/ atau https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset.

### Dataset Ratings (ratings.csv)
- **Jumlah Data**: 100.836 baris, 4 kolom
- **Kondisi Data**:
  - Tidak ada *missing values* atau duplikat.
  - Rating berkisar dari 0.5 hingga 5.0 (interval 0.5), dengan rata-rata sekitar 3.5.
- **Fitur**:
  - `userId`: ID unik pengguna (1 hingga 610).
  - `movieId`: ID unik film.
  - `rating`: Nilai rating (0.5 hingga 5.0).
  - `timestamp`: Waktu rating diberikan (Unix timestamp).

### Dataset Movies (movies.csv)
- **Jumlah Data**: 9.742 baris, 3 kolom
- **Kondisi Data**:
  - Tidak ada *missing values* atau duplikat.
  - Tahun rilis film berkisar dari 1902 hingga 2018.
- **Fitur**:
  - `movieId`: ID unik film.
  - `title`: Judul film dengan tahun rilis.
  - `genres`: Genre film, dipisahkan dengan tanda '|'.

### Dataset Tags (tags.csv)
- **Jumlah Data**: 3.683 baris, 4 kolom
- **Kondisi Data**: Tidak ada *missing values* atau duplikat.
- **Fitur**:
  - `userId`: ID unik pengguna.
  - `movieId`: ID unik film.
  - `tag`: Tag yang diberikan pengguna pada film.
  - `timestamp`: Waktu tag diberikan (Unix timestamp).

### Dataset Links (links.csv)
- **Jumlah Data**: 9.742 baris, 3 kolom
- **Kondisi Data**:
  - Tidak ada *missing values* di `movieId` dan `imdbId`, tetapi 8 *missing values* di `tmdbId`.
  - Tidak ada duplikat.
- **Fitur**:
  - `movieId`: ID unik film.
  - `imdbId`: ID film di IMDb.
  - `tmdbId`: ID film di The Movie Database (TMDb).

### Exploratory Data Analysis (EDA)
Berikut adalah insight dari analisis eksplorasi data:

1. **Distribusi Rating**:
   - Rating 4.0 paling sering (sekitar 26.818 kali), diikuti oleh 3.0 (20.047 kali) dan 5.0 (13.211 kali).
   - Rating rendah (0.5, 1.0, 1.5) jarang, menunjukkan bias seleksi pengguna yang cenderung menonton film yang mereka sukai.

2. **Tahun Rilis Film**:
   - Jumlah film meningkat signifikan sejak 1990-an, dengan puncak pada 2000-2010.
   - Penurunan setelah 2010 mungkin karena keterlambatan pembaruan database.

3. **Distribusi Genre**:
   - Drama (4.000+ film) dan Comedy (3.500+ film) adalah genre paling dominan.
   - Genre seperti Film-Noir, IMAX, dan Documentary lebih jarang.

4. **Aktivitas Pengguna**:
   - Rata-rata pengguna memberikan 165 rating, tetapi distribusi *right-skewed* (mayoritas 20-200 rating, beberapa pengguna >500 rating).

5. **Popularitas Film**:
   - Film seperti *Forrest Gump (1994)*, *The Shawshank Redemption (1994)*, dan *Pulp Fiction (1994)* memiliki rating terbanyak.
   - Film klasik dan *blockbuster* mendominasi popularitas.

## Data Preparation

Tahapan *data preparation* dilakukan untuk mempersiapkan data agar sesuai dengan kebutuhan kedua pendekatan sistem rekomendasi: **Content-Based Filtering** dan **Collaborative Filtering**. Berikut adalah penjelasan rinci untuk setiap pendekatan, termasuk alasan setiap langkah.

### Data Preparation untuk Content-Based Filtering
1. **Transformasi Data Genre**:
   - **Proses**: Kolom `genres` diubah menjadi vektor numerik menggunakan *TF-IDF Vectorizer*. Genre yang dipisahkan dengan '|' diubah menjadi string dengan spasi untuk diproses oleh *TF-IDF*.
   - **Alasan**: Algoritma machine learning memerlukan data numerik. *TF-IDF* memberikan bobot lebih tinggi pada genre yang unik dan relevan, mengurangi dampak genre umum seperti Drama.

2. **Pembuatan Reverse Mapping**:
   - **Proses**: Membuat *mapping* dari judul film ke indeks dalam *dataframe* untuk mempermudah pencarian film berdasarkan judul.
   - **Alasan**: Memungkinkan akses cepat ke indeks film saat menghasilkan rekomendasi, meningkatkan efisiensi fungsi rekomendasi.

### Data Preparation untuk Collaborative Filtering
1. **Konversi Timestamp**:
   - **Proses**: Mengubah kolom `timestamp` (Unix) menjadi format *datetime* menggunakan `pd.to_datetime`.
   - **Alasan**: Memungkinkan analisis temporal, meskipun dalam proyek ini tidak digunakan secara langsung, konversi ini memastikan fleksibilitas untuk analisis waktu di masa depan.

2. **Penggabungan Dataset**:
   - **Proses**: Menggabungkan *dataset* `ratings` dan `movies` berdasarkan `movieId` untuk mendapatkan informasi lengkap (judul, genre, rating).
   - **Alasan**: Memudahkan akses ke informasi film saat menghasilkan rekomendasi, memastikan hasil rekomendasi menyertakan detail seperti judul dan genre.

3. **Pembuatan Matriks User-Item**:
   - **Proses**: Membuat *pivot table* dengan `userId` sebagai indeks, `movieId` sebagai kolom, dan `rating` sebagai nilai, mengisi nilai kosong dengan 0.
   - **Alasan**: Matriks ini diperlukan untuk *Collaborative Filtering* karena merepresentasikan hubungan antara pengguna dan film. Nilai 0 menunjukkan film yang belum diberi rating.

4. **Normalisasi Rating**:
   - **Proses**: Mengurangkan rating setiap pengguna dengan rata-rata rating mereka untuk menghasilkan matriks *demeaned* (`R_demeaned`).
   - **Alasan**: Mengatasi bias pengguna (misalnya, pengguna yang cenderung memberi rating tinggi atau rendah). Normalisasi memastikan rating relatif terhadap preferensi pengguna.

5. **Data Splitting**:
   - **Proses**: Membagi *dataset* `ratings` menjadi 80% data pelatihan (*training*) dan 20% data pengujian (*testing*) menggunakan `train_test_split` dengan `random_state=42`.
   - **Alasan**: Memungkinkan evaluasi model pada data yang tidak digunakan saat pelatihan, memastikan generalisasi model. Proporsi 80:20 adalah standar untuk menjaga keseimbangan antara pelatihan dan pengujian.

## Modeling

Proyek ini mengimplementasikan dua pendekatan sistem rekomendasi: **Content-Based Filtering** dan **Collaborative Filtering**. Berikut adalah penjelasan rinci untuk masing-masing pendekatan, termasuk kelebihan, kekurangan, dan hasil *Top-10 Recommendations*.

### Content-Based Filtering
**Pendekatan**: Merekomendasikan film berdasarkan kemiripan genre dengan film yang telah disukai pengguna.

#### Proses
1. **TF-IDF Vectorization**: Mengubah genre film menjadi vektor numerik menggunakan *TF-IDF Vectorizer*.
2. **Cosine Similarity**: Menghitung kemiripan antar film berdasarkan vektor *TF-IDF* menggunakan formula:

   **Cosine Similarity = (A · B) / (||A|| × ||B||)**
   
   Dimana:
   - A dan B adalah vektor TF-IDF dari dua film
   - A · B adalah produk dot (dot product)
   - ||A|| dan ||B|| adalah norma Euclidean dari vektor A dan B

3. **Fungsi Rekomendasi**: Mengambil judul film sebagai input, mengurutkan film berdasarkan skor *cosine similarity*, dan mengembalikan 10 film teratas (tidak termasuk film input).

#### Hasil Top-10 Recommendations
Untuk film **"Toy Story (1995)"** (genre: Adventure|Animation|Children|Comedy|Fantasy), hasil rekomendasi adalah:

| Title | Genres | Similarity Score |
|-------|--------|------------------|
| Antz (1998) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Toy Story 2 (1999) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Adventures of Rocky and Bullwinkle, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Emperor's New Groove, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Monsters, Inc. (2001) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Wild, The (2006) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Shrek the Third (2007) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Tale of Despereaux, The (2008) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Asterix and the Vikings (2006) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |
| Turbo (2013) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0 |

**Kelebihan**:
- Efektif untuk *cold start problem* karena hanya bergantung pada fitur film.
- Sederhana dan cepat untuk dataset dengan fitur terbatas seperti genre.

**Kekurangan**:
- Terbatas pada fitur genre, tidak menangkap aspek lain seperti aktor atau sinopsis.
- Tidak mempertimbangkan preferensi pengguna di luar film yang telah ditonton.

### Collaborative Filtering
**Pendekatan**: Merekomendasikan film berdasarkan pola rating pengguna lain yang serupa, menggunakan *Singular Value Decomposition (SVD)*.

#### Proses
1. **Matriks User-Item**: Membuat matriks rating dengan pengguna sebagai baris dan film sebagai kolom.
2. **Normalisasi**: Mengurangkan rata-rata rating per pengguna untuk menghilangkan bias.
3. **SVD**: Menerapkan SVD pada matriks *demeaned* untuk mengurangi dimensi:

   **R ≈ U × Σ × V^T**
   
   Dimana:
   - R adalah matriks rating
   - U adalah matriks pengguna (user matrix)
   - Σ adalah matriks diagonal nilai singular
   - V^T adalah transpose matriks film (item matrix)
   - Parameter k=50 digunakan untuk jumlah faktor laten

4. **Prediksi Rating**: Mengalikan kembali U, Σ, dan V^T, lalu menambahkan rata-rata rating pengguna:

   **Prediksi Rating = (U × Σ × V^T) + Rata-rata Rating Pengguna**

5. **Fungsi Rekomendasi**: Mengambil ID pengguna, mengidentifikasi film yang belum ditonton, dan merekomendasikan 10 film dengan prediksi rating tertinggi.

#### Hyperparameter Tuning
- Parameter k (jumlah faktor laten) diuji dengan nilai 20, 50, dan 100.
- Nilai k=50 dipilih karena memberikan keseimbangan antara akurasi (RMSE lebih rendah) dan efisiensi komputasi dibandingkan k=20 (akurasi rendah) dan k=100 (risiko *overfitting*).

#### Hasil Top-10 Recommendations
Untuk pengguna dengan `userId=1`, hasil rekomendasi adalah:

| Title | Genres | Predicted Rating |
|-------|--------|------------------|
| Die Hard (1988) | Action\|Crime\|Thriller | 4.024307 |
| Godfather: Part II, The (1974) | Crime\|Drama | 3.324815 |
| Jaws (1975) | Action\|Horror | 3.304728 |
| Godfather, The (1972) | Crime\|Drama | 2.891690 |
| Breakfast Club, The (1985) | Comedy\|Drama | 2.870832 |
| Stand by Me (1986) | Adventure\|Drama | 2.786815 |
| Christmas Story, A (1983) | Children\|Comedy | 2.587995 |
| Lady and the Tramp (1955) | Animation\|Children\|Comedy\|Romance | 2.442516 |
| Snatch (2000) | Comedy\|Crime\|Thriller | 2.395703 |
| Little Mermaid, The (1989) | Animation\|Children\|Comedy\|Musical\|Romance | 2.383887 |

**Kelebihan**:
- Menangkap pola preferensi kompleks berdasarkan rating pengguna lain.
- Memberikan rekomendasi yang lebih personal dan beragam.

**Kekurangan**:
- Memerlukan data rating yang cukup, rentan terhadap *cold start problem*.
- Komputasi intensif untuk matriks besar.

## Evaluation

### Content-Based Filtering
**Metrik Evaluasi**: *Precision@10*, yang mengukur proporsi film rekomendasi yang relevan (memiliki setidaknya satu genre sama dengan film referensi) di antara 10 rekomendasi teratas:

**Precision@10 = (Jumlah film relevan dalam Top-10) / 10**

**Penjelasan Metrik**: *Precision@10* menghitung akurasi rekomendasi berdasarkan kesamaan genre. Film dianggap relevan jika memiliki setidaknya satu genre yang sama dengan film input, yang sesuai dengan tujuan *Content-Based Filtering* untuk merekomendasikan film dengan karakteristik serupa.

**Hasil**: Untuk film seperti *Toy Story (1995)*, *The Dark Knight (2008)*, dan lainnya, *Precision@10* rata-rata adalah **1.0**, menunjukkan bahwa semua rekomendasi memiliki setidaknya satu genre yang sama. Namun, definisi "relevan" ini cukup longgar; metrik seperti *Jaccard similarity* dapat digunakan untuk evaluasi yang lebih ketat.

### Collaborative Filtering
**Metrik Evaluasi**:
1. **Root Mean Squared Error (RMSE)**:

   **RMSE = √[(1/N) × Σ(y_i - ŷ_i)²]**
   
   Dimana:
   - y_i adalah rating sebenarnya
   - ŷ_i adalah rating prediksi
   - N adalah jumlah data pengujian

2. **Mean Absolute Error (MAE)**:

   **MAE = (1/N) × Σ|y_i - ŷ_i|**
   
   Dimana:
   - y_i adalah rating sebenarnya
   - ŷ_i adalah rating prediksi
   - N adalah jumlah data pengujian

**Penjelasan Metrik**:
- **RMSE** mengukur rata-rata kuadrat error antara rating sebenarnya dan prediksi, memberikan penalti lebih besar pada error yang besar.
- **MAE** mengukur rata-rata error absolut, memberikan gambaran langsung tentang deviasi prediksi.

**Hasil**:
- **Sebelum Tuning**: Dengan k=20, RMSE = 3.20, MAE = 3.00.
- **Setelah Tuning**: Dengan k=50, RMSE = **3.1673**, MAE = **2.9575** (sesuai output notebook). Tuning ke k=100 menghasilkan RMSE sedikit lebih tinggi (3.18), menunjukkan *overfitting*.
- **Interpretasi**: RMSE 3.1673 dan MAE 2.9575 menunjukkan rata-rata error prediksi sekitar 3 poin pada skala 0.5-5.0, yang kurang ideal. Faktor seperti *sparsity* matriks rating dan *cold start problem* berkontribusi pada error ini.

**Perbaikan Potensial**:
- Meningkatkan jumlah faktor laten atau mencoba regularisasi untuk SVD.
- Menggunakan algoritma lain seperti *Neural Collaborative Filtering*.
- Mengatasi *sparsity* dengan teknik *data imputation*.

## Conclusion

Proyek ini berhasil mengimplementasikan dua pendekatan sistem rekomendasi film:

1. **Content-Based Filtering**:
   - **Hasil**: *Precision@10* = 1.0, menunjukkan rekomendasi sangat relevan berdasarkan genre.
   - **Kelebihan**: Efektif untuk *cold start problem* dan sederhana untuk fitur terbatas.
   - **Kekurangan**: Terbatas pada genre, tidak menangkap preferensi kompleks pengguna.
   - **Use Case**: Cocok untuk pengguna baru atau film dengan sedikit rating.

2. **Collaborative Filtering**:
   - **Hasil**: RMSE = 3.1673, MAE = 2.9575, menunjukkan performa moderat.
   - **Kelebihan**: Memberikan rekomendasi personal berdasarkan pola rating pengguna lain.
   - **Kekurangan**: Rentan terhadap *cold start problem* dan membutuhkan data rating yang cukup.
   - **Use Case**: Ideal untuk platform dengan banyak data rating.

**Rekomendasi Pengembangan**:
- **Hybrid Approach**: Menggabungkan *Content-Based* dan *Collaborative Filtering* untuk memanfaatkan kelebihan keduanya.
- **Feature Enrichment**: Menambahkan fitur seperti aktor, sutradara, atau sinopsis untuk *Content-Based Filtering*.
- **Deep Learning**: Menggunakan *Neural Collaborative Filtering* untuk menangkap pola yang lebih kompleks.

Dengan pendekatan ini, sistem rekomendasi dapat disesuaikan dengan kebutuhan platform dan karakteristik data, meningkatkan pengalaman pengguna secara signifikan.