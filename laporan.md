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

Dataset MovieLens Small Latest adalah dataset yang berisi rating film dari pengguna MovieLens. Dataset ini dikeluarkan oleh GroupLens Research dan merupakan versi kecil dari dataset MovieLens yang lebih besar. Dataset ini dapat diunduh dari [GroupLens](https://grouplens.org/datasets/movielens/latest/) atau melalui [Kaggle](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset).

Dataset ini terdiri dari 100.836 rating dan 3.683 tag yang diberikan pada 9.742 film oleh 610 pengguna. Data ini dibuat antara 29 Maret 1996 dan 24 September 2018. Dataset ini didesain untuk tujuan pendidikan dan penelitian dalam bidang sistem rekomendasi.

Dataset ini terdiri dari beberapa file:

1. **ratings.csv**: Berisi rating film dari pengguna dengan variabel:
   - `userId`: ID unik untuk setiap pengguna (1 sampai 610)
   - `movieId`: ID unik untuk setiap film
   - `rating`: Rating yang diberikan oleh pengguna (skala 0.5 sampai 5.0 dengan interval 0.5)
   - `timestamp`: Waktu rating diberikan (dalam format Unix timestamp)

2. **movies.csv**: Berisi informasi tentang film dengan variabel:
   - `movieId`: ID unik untuk setiap film
   - `title`: Judul film yang disertai dengan tahun rilis dalam tanda kurung
   - `genres`: Genre film yang dipisahkan dengan karakter '|'

3. **tags.csv**: Berisi tag yang diberikan pengguna pada film dengan variabel:
   - `userId`: ID unik untuk setiap pengguna
   - `movieId`: ID unik untuk setiap film
   - `tag`: Tag yang diberikan oleh pengguna pada film tertentu
   - `timestamp`: Waktu tag diberikan (dalam format Unix timestamp)

4. **links.csv**: Berisi tautan ke database film lain dengan variabel:
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

Timestamp dalam format Unix dikonversi menjadi format datetime untuk memudahkan analisis berdasarkan waktu.

```python
# Mengkonversi timestamp menjadi format datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
tags['timestamp'] = pd.to_datetime(tags['timestamp'], unit='s')
```

Konversi ini diperlukan untuk memudahkan analisis temporal, seperti melihat tren rating berdasarkan waktu atau menganalisis pola pengguna dalam memberikan rating pada periode tertentu.

### 2. Penggabungan Dataset

Dataset ratings dan movies digabungkan berdasarkan kolom movieId untuk mendapatkan informasi lengkap tentang film dan rating-nya.

```python
# Menggabungkan dataset ratings dan movies berdasarkan movieId
ratings_movies = pd.merge(ratings, movies, on='movieId')
```

Penggabungan ini diperlukan untuk mengakses informasi film (seperti judul dan genre) bersama dengan data rating dalam satu dataset, yang memudahkan analisis dan pengembangan model.

### 3. Transformasi Data untuk Content-Based Filtering

Untuk model Content-Based Filtering, fitur genre film ditransformasi menjadi representasi vektor menggunakan TF-IDF Vectorizer.

```python
# Membuat TF-IDF matrix dari genre film
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')  # Pastikan tidak ada nilai NaN
tfidf_matrix = tfidf.fit_transform(movies['genres'].str.replace('|', ' '))
```

Transformasi ini diperlukan karena algoritma machine learning bekerja dengan data numerik, sementara genre film adalah data tekstual. TF-IDF mengubah data kategorik menjadi representasi numerik yang menangkap makna semantik dari genre film.

### 4. Perhitungan Cosine Similarity

Cosine similarity dihitung antara semua pasangan film berdasarkan representasi TF-IDF dari genre mereka.

```python
# Menghitung cosine similarity antara film berdasarkan genre
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

Perhitungan similarity ini diperlukan untuk mengukur kemiripan antar film berdasarkan genre mereka, yang akan digunakan dalam model Content-Based Filtering untuk menemukan film yang mirip.

### 5. Pembuatan Reverse Mapping

Untuk mempermudah pencarian film, dibuat mapping dari judul film ke indeksnya dalam dataframe.

```python
# Membuat reverse mapping dari judul film ke indeks
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
```

Mapping ini memungkinkan kita untuk dengan cepat menemukan indeks film berdasarkan judulnya, yang diperlukan dalam fungsi rekomendasi.

### 6. Normalisasi Rating untuk Collaborative Filtering

Untuk model Collaborative Filtering, rating pengguna dinormalisasi dengan mengurangkan nilai rata-rata rating pengguna tersebut.

```python
# Menghitung rata-rata rating untuk setiap user
user_ratings_mean = np.mean(R, axis=1)

# Mengurangkan rating dengan rata-rata untuk mendapatkan rating relatif
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
```

Normalisasi ini penting untuk mengatasi bias pengguna, di mana beberapa pengguna cenderung memberikan rating lebih tinggi atau lebih rendah secara konsisten. Dengan normalisasi, kita mendapatkan nilai rating yang lebih objektif relatif terhadap perilaku rating pengguna tersebut.

## Modeling

Dalam proyek ini, saya mengimplementasikan dua pendekatan sistem rekomendasi: Content-Based Filtering dan Collaborative Filtering.

### 1. Content-Based Filtering

Content-Based Filtering merekomendasikan item berdasarkan kemiripan fitur dengan item yang disukai pengguna sebelumnya. Dalam konteks film, fitur yang digunakan adalah genre film.

#### Implementasi Model

1. **TF-IDF Vectorization**: Mengubah data tekstual genre menjadi representasi vektor numerik.
2. **Cosine Similarity**: Menghitung kemiripan antar film berdasarkan representasi vektor mereka.
3. **Fungsi Rekomendasi**: Mengambil judul film sebagai input dan mengembalikan film-film yang memiliki kemiripan tertinggi.

```python
def get_recommendations(title, cosine_sim=cosine_sim, movies=movies, indices=indices):
    """
    Fungsi untuk mendapatkan rekomendasi film berdasarkan kemiripan genre

    Args:
        title (str): Judul film
        cosine_sim (numpy.ndarray): Matrix cosine similarity
        movies (pandas.DataFrame): DataFrame berisi informasi film
        indices (pandas.Series): Mapping dari judul film ke indeks

    Returns:
        pandas.DataFrame: DataFrame berisi rekomendasi film
    """
    # Dapatkan indeks film
    try:
        idx = indices[title]
    except:
        return pd.DataFrame({"Judul": ["Film tidak ditemukan"], "Genre": [""]})

    # Dapatkan skor kemiripan untuk semua film
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Urutkan film berdasarkan skor kemiripan
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Dapatkan 10 film yang paling mirip (tidak termasuk film itu sendiri)
    sim_scores = sim_scores[1:11]

    # Dapatkan indeks film
    movie_indices = [i[0] for i in sim_scores]

    # Kembalikan 10 film yang paling mirip beserta skor similaritynya
    result_df = movies.iloc[movie_indices][['title', 'genres']].copy()
    result_df['similarity_score'] = [i[1] for i in sim_scores]

    return result_df
```

#### Hasil Rekomendasi

Berikut contoh rekomendasi untuk film "Toy Story (1995)":

```plaintext
      title                                                genres                                        similarity_score
1706  Antz (1998)                                          Adventure|Animation|Children|Comedy|Fantasy    1.0
2355  Toy Story 2 (1999)                                   Adventure|Animation|Children|Comedy|Fantasy    1.0
2809  Adventures of Rocky and Bullwinkle, The (2000)       Adventure|Animation|Children|Comedy|Fantasy    1.0
3000  Emperor's New Groove, The (2000)                     Adventure|Animation|Children|Comedy|Fantasy    1.0
3568  Monsters, Inc. (2001)                                Adventure|Animation|Children|Comedy|Fantasy    1.0
6194  Wild, The (2006)                                     Adventure|Animation|Children|Comedy|Fantasy    1.0
6486  Shrek the Third (2007)                               Adventure|Animation|Children|Comedy|Fantasy    1.0
6948  Tale of Despereaux, The (2008)                       Adventure|Animation|Children|Comedy|Fantasy    1.0
7760  Asterix and the Vikings (Astérix et les Vikings...)  Adventure|Animation|Children|Comedy|Fantasy    1.0
8219  Turbo (2013)                                         Adventure|Animation|Children|Comedy|Fantasy    1.0
```

Dari hasil rekomendasi, terlihat bahwa model berhasil merekomendasikan film-film dengan genre yang sangat mirip dengan "Toy Story (1995)". Semua film yang direkomendasikan memiliki genre Adventure, Animation, Children, Comedy, dan Fantasy, dengan skor kemiripan 1.0, menunjukkan kecocokan sempurna dengan genre film asli. Rekomendasi ini mencakup film dari berbagai tahun, termasuk sekuel seperti "Toy Story 2" dan film animasi populer lainnya seperti "Antz" dan "Monsters, Inc.".

### 2. Collaborative Filtering

Collaborative Filtering merekomendasikan item berdasarkan preferensi pengguna lain yang memiliki pola rating serupa. Pendekatan ini menggunakan teknik Singular Value Decomposition (SVD) untuk mengungkap pola tersembunyi dalam data rating.

#### Implementasi Model

1. **Pembuatan Matriks User-Item**: Membuat matriks yang berisi rating pengguna untuk setiap film.
2. **Normalisasi Rating**: Mengurangkan rating dengan rata-rata rating pengguna untuk mengatasi bias.
3. **Singular Value Decomposition (SVD)**: Menerapkan SVD untuk mengungkap pola tersembunyi dalam data rating.
4. **Prediksi Rating**: Menggunakan hasil SVD untuk memprediksi rating pengguna terhadap film yang belum mereka tonton.
5. **Rekomendasi Film**: Merekomendasikan film dengan prediksi rating tertinggi.

```python
# Membuat pivot table dari data rating
user_movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Ubah pivot table menjadi numpy array
R = user_movie_ratings.to_numpy()

# Menghitung rata-rata rating untuk setiap user
user_ratings_mean = np.mean(R, axis=1)

# Mengurangkan rating dengan rata-rata untuk mendapatkan rating relatif
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Menerapkan SVD pada matrix R_demeaned
U, sigma, Vt = svds(R_demeaned, k=50)

# Mengubah sigma dari array menjadi diagonal matrix
sigma = np.diag(sigma)

# Prediksi rating dengan SVD
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# Mengkonversi hasil prediksi kembali menjadi DataFrame
preds_df = pd.DataFrame(all_user_predicted_ratings,
                       index=user_movie_ratings.index,
                       columns=user_movie_ratings.columns)
```

Fungsi untuk mendapatkan rekomendasi film untuk pengguna tertentu:

```python
def recommend_movies_for_user(user_id, preds_df, movies_df, ratings_df, num_recommendations=10):
    """
    Fungsi untuk mendapatkan rekomendasi film untuk pengguna tertentu

    Args:
        user_id (int): ID pengguna
        preds_df (pandas.DataFrame): DataFrame berisi prediksi rating
        movies_df (pandas.DataFrame): DataFrame berisi informasi film
        ratings_df (pandas.DataFrame): DataFrame berisi rating pengguna
        num_recommendations (int): Jumlah rekomendasi yang diinginkan

    Returns:
        pandas.DataFrame: DataFrame berisi rekomendasi film
    """
    # Dapatkan film yang belum ditonton oleh pengguna
    user_data = ratings_df[ratings_df.userId == user_id]
    user_watched_movies = user_data.movieId.unique()
    
    # Dapatkan film yang belum ditonton
    movies_not_watched = movies_df[~movies_df['movieId'].isin(user_watched_movies)]
    
    # Dapatkan prediksi rating untuk film yang belum ditonton
    user_predictions = preds_df.loc[user_id, movies_not_watched['movieId']]
    
    # Gabungkan prediksi dengan informasi film
    recommendations = pd.DataFrame({'movieId': user_predictions.index,
                                   'predicted_rating': user_predictions.values})
    recommendations = recommendations.sort_values('predicted_rating', ascending=False)
    
    # Gabungkan dengan informasi film
    recommendations = recommendations.merge(movies_df, on='movieId')
    
    return recommendations.head(num_recommendations)
```

#### Hasil Rekomendasi

Berikut contoh rekomendasi untuk pengguna dengan ID 1:

```plaintext
   movieId  predicted_rating                           title                                        genres           year                           genres_list
0  1036     4.024307          Die Hard (1988)                           Action|Crime|Thriller         1988.0         [Action, Crime, Thriller]
1  1221     3.324815          Godfather: Part II, The (1974)           Crime|Drama                    1974.0         [Crime, Drama]
2  1387     3.304728          Jaws (1975)                              Action|Horror                  1975.0         [Action, Horror]
3  858      2.891690          Godfather, The (1972)                    Crime|Drama                    1972.0         [Crime, Drama]
4  1968     2.870832          Breakfast Club, The (1985)               Comedy|Drama                   1985.0         [Comedy, Drama]
5  1259     2.786815          Stand by Me (1986)                       Adventure|Drama                1986.0         [Adventure, Drama]
6  2804     2.587995          Christmas Story, A (1983)                Children|Comedy                1983.0         [Children, Comedy]
7  2080     2.442516          Lady and the Tramp (1955)                Animation|Children|Comedy|Romance  1955.0         [Animation, Children, Comedy, Romance]
8  4011     2.395703          Snatch (2000)                            Comedy|Crime|Thriller         2000.0         [Comedy, Crime, Thriller]
9  2081     2.383887          Little Mermaid, The (1989)               Animation|Children|Comedy|Musical|Romance 1989.0  [Animation, Children, Comedy, Musical, Romance]
```

Dari hasil rekomendasi, terlihat bahwa model Collaborative Filtering berhasil merekomendasikan film-film dengan prediksi rating tinggi untuk pengguna 1. Rekomendasi ini didasarkan pada pola rating pengguna lain yang memiliki preferensi serupa dengan pengguna 1. Film-film yang direkomendasikan mencakup berbagai genre seperti Action, Crime, Thriller, Drama, Comedy, dan Animation, yang mungkin mencerminkan preferensi pengguna 1 berdasarkan film-film yang telah mereka tonton dan beri rating sebelumnya. Film dengan rating prediksi tertinggi adalah "Die Hard (1988)" dengan skor 4.02, diikuti oleh film klasik seperti "Godfather: Part II" dan "Jaws".

## Evaluation

Untuk mengevaluasi kedua model rekomendasi yang telah diimplementasikan, saya menggunakan metrik evaluasi yang sesuai dengan karakteristik masing-masing model.

### 1. Evaluasi Content-Based Filtering

Untuk Content-Based Filtering, saya menggunakan **Precision@K** (dalam hal ini Precision@10) sebagai metrik evaluasi. Precision@K mengukur proporsi item yang relevan di antara top K rekomendasi.

#### Formula:

```
Precision@K = (Jumlah rekomendasi yang relevan di antara K rekomendasi) / K
```

Dalam konteks rekomendasi film berdasarkan genre, "relevan" didefinisikan sebagai film yang memiliki setidaknya satu genre yang sama dengan film referensi.

```python
def evaluate_content_based_recommender(test_movies, indices, cosine_sim, movies):
    """
    Fungsi untuk mengevaluasi sistem rekomendasi berbasis konten

    Args:
        test_movies (list): Daftar film untuk diuji
        indices (pandas.Series): Mapping dari judul film ke indeks
        cosine_sim (numpy.ndarray): Matrix cosine similarity
        movies (pandas.DataFrame): DataFrame berisi informasi film

    Returns:
        float: Precision@10 rata-rata
    """
    precision_list = []

    for title in test_movies:
        # Dapatkan indeks film
        try:
            idx = indices[title]
        except:
            continue

        # Dapatkan genre film yang diuji
        target_genres = set(movies.iloc[idx]['genres'].split('|'))

        # Dapatkan skor kemiripan untuk semua film
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Urutkan film berdasarkan skor kemiripan
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Dapatkan 10 film yang paling mirip
        sim_scores = sim_scores[1:11]

        # Dapatkan indeks film
        movie_indices = [i[0] for i in sim_scores]

        # Hitung presisi (berapa banyak genre yang cocok)
        relevant = 0
        for i in movie_indices:
            recommended_genres = set(movies.iloc[i]['genres'].split('|'))
            # Setidaknya satu genre yang sama
            if len(target_genres.intersection(recommended_genres)) > 0:
                relevant += 1

        precision = relevant / 10
        precision_list.append(precision)

    return sum(precision_list) / len(precision_list)
```

#### Hasil Evaluasi:

Precision@10 dihitung untuk sejumlah film populer:

```python
test_movies = [
    'Toy Story (1995)',
    'The Dark Knight (2008)',
    'Pulp Fiction (1994)',
    'Forrest Gump (1994)',
    'The Matrix (1999)',
    'Titanic (1997)',
    'The Shawshank Redemption (1994)',
    'Avatar (2009)',
    'Inception (2010)',
    'The Godfather (1972)'
]

precision_at_10 = evaluate_content_based_recommender(test_movies, indices, cosine_sim, movies)
print(f'Precision@10 rata-rata: {precision_at_10:.4f}')
# Output: Precision@10 rata-rata: 1.0000
```

Model Content-Based Filtering mencapai Precision@10 rata-rata sebesar 1.0, menunjukkan bahwa 100% dari rekomendasi memiliki setidaknya satu genre yang sama dengan film referensi. Ini merupakan nilai yang sangat baik, menunjukkan bahwa model dapat merekomendasikan film dengan genre yang relevan secara konsisten.

### 2. Evaluasi Collaborative Filtering

Untuk Collaborative Filtering, saya menggunakan **Root Mean Squared Error (RMSE)** dan **Mean Absolute Error (MAE)** sebagai metrik evaluasi. Kedua metrik ini mengukur akurasi prediksi rating model.

#### Formula:

```
RMSE = √(Σ(y_true - y_pred)² / n)
MAE = Σ|y_true - y_pred| / n
```

Dimana:
- y_true adalah rating sebenarnya
- y_pred adalah rating yang diprediksi
- n adalah jumlah prediksi

```python
def rmse(y_true, y_pred):
    """
    Fungsi untuk menghitung RMSE

    Args:
        y_true (array): Nilai sebenarnya
        y_pred (array): Nilai prediksi

    Returns:
        float: RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """
    Fungsi untuk menghitung MAE

    Args:
        y_true (array): Nilai sebenarnya
        y_pred (array): Nilai prediksi

    Returns:
        float: MAE
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
```

#### Hasil Evaluasi: 

Evaluasi dilakukan dengan membagi dataset menjadi data training (80%) dan data testing (20%), kemudian mengevaluasi akurasi prediksi rating pada data testing:

```python
# Membagi dataset untuk evaluasi
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Melatih model pada data training
# [kode training model collaborative filtering dengan SVD]

# Membuat prediksi untuk data test
test_rmse_list = []

for _, row in test.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    rating = row['rating']

    # Pastikan user_id dan movie_id ada dalam preds_df_train
    if user_id in preds_df_train.index and movie_id in preds_df_train.columns:
        pred_rating = preds_df_train.loc[user_id, movie_id]
        test_rmse_list.append((rating, pred_rating))

# Menghitung RMSE
y_true = [x[0] for x in test_rmse_list]
y_pred = [x[1] for x in test_rmse_list]

rmse_score = rmse(y_true, y_pred)
mae_score = mae(y_true, y_pred)
print(f'RMSE: {rmse_score:.4f}')
print(f'MAE: {mae_score:.4f}')
# Output: RMSE: 3.1673, MAE: 2.9575
```

Model Collaborative Filtering mencapai RMSE sekitar 3.17 dan MAE sekitar 2.96. Ini berarti:

1. RMSE 3.17: Rata-rata, prediksi rating model menyimpang sekitar 3.2 poin dari rating sebenarnya (pada skala 0.5-5.0).
2. MAE 2.96: Rata-rata, prediksi rating menyimpang sekitar 3.0 poin dari rating sebenarnya.

Nilai RMSE dan MAE yang cukup tinggi ini menunjukkan bahwa model masih memiliki ruang untuk perbaikan. Beberapa faktor yang mungkin menyebabkan nilai error yang tinggi:

1. Sparsitas data: Banyak pengguna hanya memberikan rating pada sebagian kecil film
2. Cold start problem: Sulit memprediksi rating untuk pengguna baru atau film baru
3. Kompleksitas preferensi: Preferensi pengguna terhadap film dipengaruhi oleh banyak faktor selain pola rating

### 3. Interpretasi dan Perbandingan Hasil Evaluasi

Berdasarkan evaluasi yang telah dilakukan, kedua model menunjukkan performa yang berbeda dalam tugas masing-masing:

1. **Content-Based Filtering**:
   - Precision@10: 1.0 (100% rekomendasi memiliki genre yang relevan)
   - Kelebihan: Sangat baik dalam merekomendasikan film dengan karakteristik serupa
   - Keterbatasan: Tidak dapat menangkap preferensi pengguna di luar fitur yang dimodelkan (hanya genre)

2. **Collaborative Filtering**:
   - RMSE: 3.17, MAE: 2.96
   - Kelebihan: Dapat menangkap preferensi pengguna yang kompleks dan tersembunyi
   - Keterbatasan: Membutuhkan data rating yang cukup dan menghadapi cold start problem

Kedua nilai evaluasi ini tidak dapat dibandingkan secara langsung karena mengukur aspek yang berbeda dari sistem rekomendasi. Content-Based Filtering dioptimalkan untuk relevansi konten, sementara Collaborative Filtering dioptimalkan untuk akurasi prediksi rating.

Untuk penggunaan praktis, pendekatan hibrid yang mengkombinasikan kedua metode ini dapat memberikan hasil terbaik:
- Menggunakan Content-Based Filtering untuk pengguna baru atau film baru (mengatasi cold start problem)
- Menggunakan Collaborative Filtering untuk pengguna yang sudah memiliki data rating yang cukup (memanfaatkan pola preferensi kompleks)

## Kesimpulan

Proyek ini telah berhasil mengimplementasikan dan mengevaluasi dua pendekatan sistem rekomendasi film: Content-Based Filtering dan Collaborative Filtering. Berdasarkan hasil yang diperoleh, dapat disimpulkan bahwa:

1. **Content-Based Filtering** sangat efektif dalam merekomendasikan film berdasarkan kemiripan genre, dengan Precision@10 mencapai 1.0. Pendekatan ini cocok untuk situasi cold start dan ketika transparansi rekomendasi penting.

2. **Collaborative Filtering** memberikan rekomendasi yang dipersonalisasi berdasarkan pola rating pengguna, meskipun dengan RMSE 3.17 dan MAE 2.96 menunjukkan bahwa masih ada ruang untuk perbaikan. Pendekatan ini lebih baik dalam menangkap preferensi kompleks yang tidak terlihat dari metadata film.

3. **Keterbatasan** dari kedua pendekatan telah diidentifikasi: Content-Based Filtering terbatas pada fitur yang dimodelkan, sementara Collaborative Filtering membutuhkan data rating yang cukup dan menghadapi cold start problem.

4. **Rekomendasi untuk pengembangan** lebih lanjut meliputi implementasi sistem rekomendasi hybrid yang mengkombinasikan kekuatan kedua pendekatan, penambahan fitur film lainnya seperti aktor dan sutradara, serta penerapan teknik deep learning seperti Neural Collaborative Filtering.

Sistem rekomendasi yang diimplementasikan dalam proyek ini dapat membantu pengguna menemukan film yang sesuai dengan preferensi mereka, mengatasi masalah paradoks pilihan, dan potensial meningkatkan pengalaman pengguna pada platform film.