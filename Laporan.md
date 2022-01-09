# Laporan Proyek Machine Learning - Farkhan Hamzah Firdaus

## Domain Proyek
Industry perfilman dunia merupakan salah satu industry yang tidak terpengaruh dengan maraknya hiburan digital seperti munculnya media social, program televisi yang beragam dan game. Industry film yang terus melakukan produksi ini semakin menambah informasi film yang melimpah di internet. Kondisi ini justru membuat para penikmat film menjadi kebingungan ketika harus  memilih film kesukaannya.Sistem rekomendasi menyediakan  informasi berdasarkan interaksi pengguna dan item yang telah terekam sebelumnya. Penelitian ini akan membahas pembangunan sistem rekomendasi dengan metode pendekatan Collaborative Filtering.


## Business Understanding
### Problem Statements
* Bagaimana cara meningkatakan user experience saat mencari movie yang ingin ditonton ?
* Bagaimana cara membuat sistem rekomendasi movie menggunakan metode collaborative filtering ?

### Goals
* Meningkatakan user experience saat mencari film yang ingin ditonton.
* Dapat mengimplementasikan metode collaborative filtering untuk sistem rekomendasi movie.

### Solution statements
Dataset yang saya gunakan hanya berisi tentang rating atau hasil penilaian pengguna dan genre film, maka solusi yang sangat tepat untuk masalah ini adalah dengan menggunakan collaborative filtering.
[Collaborative Filtering](https://medium.com/@ranggaantok/bagaimana-sistem-rekomendasi-berkerja-e749dac64816): collaborative filtering adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna.
Pada collaborative filtering attribut yang digunakan bukan konten tetapi user behaviour. contohnya kita merekomendasikan suatu item berdasarkan dari riwayat rating dari user tersebut maupun user lain.
![image](https://miro.medium.com/max/335/1*O6ON-kQ34pMCYOHSr7ZebQ.png)



## Data Understanding
Untuk dataset sendiri saya ambil dari [Movie Lens Dataset](https://www.kaggle.com/aigamer/movie-lens-dataset) yang berada di platform [kaggle](https://www.kaggle.com/). Berikut adalah keterangan mengenai maksud dari variable - variable atau kolom tersebut:

* movies.csv
    * movieId : Unique Id disediakan untuk setiap Film
    * title : Nama film dengan Tahun dalam tanda kurung
    * genres : Genre pada film tersebut
* ratings.csv
    * userId : Unique Id disediakan untuk setiap Pengguna
    * movieId : Unique Id disediakan untuk setiap Film
    * rating : Penilaian pengguna terhadap film terkait
    * timestamp : Kode waktu film
* tags.csv
    * userId : Unique Id disediakan untuk setiap Pengguna
    * movieId : Unique Id disediakan untuk setiap Film
    * tag : Metadata yang dibuat pengguna tentang film. 
    * timestamp : Kode waktu film

Beriku adalah overview dari dataset tersebut setelah saya jadikan dataframe:
movie_df adalah isi dari dataset movies.csv.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/movie_df.png?token=ANXJTPI7O3CQVIUWOT6GRFDBRTXRE)

rating_df adalah isi dari dataset rating.csv yang sebelumnya telas saya hilangkan kolom timestamp nya.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/rating_df.png?token=ANXJTPLCGYYWZU2H77KJKM3BRTXRK)

tags_df adalah isi dari dataset tags.csv.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/tags_df.png?token=ANXJTPOT5PNFYFEYR7BS7P3BRTXRO)

Cek informasi di setiap dataset

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/movie_df_info.png?token=ANXJTPJCC7UK3POHPX4WGX3BSSJQE)

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/rating_df_info.png?token=ANXJTPOSQFCOWYVHOOC2SWTBSSJQI)

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/tags_df_info.png?token=ANXJTPI4L4T7OEGWLF4XWJTBSSJQO)



Cek data null Data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Cara untuk melihat apakah data ini mengandung null atau tidak adalah dengan menggunakan method dari library pandas yaitu isnull(). Berikut adalah hasil dari cek data null oleh pandas : 
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/isnullMovie.png?token=ANXJTPMRIYSNG2VGBRGC7UDBSM6UU)

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/isnullRating.png?token=ANXJTPPGKL5J7ZTGG5DB3ULBSM6VK)

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/isnullTag.png?token=ANXJTPKLZPAYWEBDWF7QIFTBSM6VO)

## Data Preparation
Untuk data preparation sendiri saya meggunakan beberapa cara. Ada 3 dataframe yang akan saya periksa dan siapkan yaitu movie_df, rating_df, dan tags_df. Berikut penjelasan beberapa teknik yang saya gunakan untuk data preparationdan:

1. Removing missing value, tahapan ini diperlukan karena dengan tidak adanya missing value akan membuat performa dalam pembuatan model menjadi lebih baik. Tahapan ini dilakukan dengan code seperti berikut: dataframe.dropna(). Kode ini berfungsi untuk menghapuskan data yang memiliki null values di dalam row setiap data.

2. Normalisasi yaitu untuk mengubah nilai kolom numerik dalam kumpulan data ke skala umum, tanpa mendistorsi perbedaan dalam rentang nilai. Proses normalisasi dilakukan dengan metode Min Max. Proses tersebut dilakukan dengan code seperti gambar di bawah ini : ![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/normalisasiRating.png?token=ANXJTPIEDZUKKPR3UMVCI63BSM63O)


## Modeling
Untuk proses pemodelan di sini saya menggunakan teknik embedding. Saya menggunakan Model [Neural Collaborative Filtering (NCF)](https://towardsdatascience.com/paper-review-neural-collaborative-filtering-explanation-implementation-ea3e031b7f96). Model Neural Collaborative Filtering (NCF) adalah jaringan saraf (neural network) yang menyediakan Collaborative Filtering berdasarkan umpan balik implisit. Secara khusus, ini memberikan rekomendasi produk berdasarkan interaksi pengguna dan item. Data pelatihan untuk model ini harus berisi urutan pasangan (ID pengguna, ID anime) yang menunjukkan bahwa pengguna yang ditentukan telah berinteraksi dengan item, misalnya, dengan memberi peringkat atau mengklik. NCF pertama kali dijelaskan oleh Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu dan Tat-Seng Chua dalam makalah [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031).

Menampilkan item movie yang user ini sukai dan tidak sebelumnya
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/ygdisukaipenggunaSebelum.png?token=ANXJTPIHL22OAL33GHWU6STBSM6F2)

Berikut merupakan langkah untuk mendapatkan list rekomendasi movie berdasarkan aktivitas user berdasarkan rate yang diberikan oleh user.

1. Mencari data movie apa saja yang telah ditonton oleh user lalu memasukkannya ke dalam dataframe yang baru
2. Lalu mencari rating terendah dari movie
3. Selanjutnya membuat top_movie_refference dengan mengurutkannya berdasarkan rating dari movie.
4. Setelah itu saya membuat dataframe baru (user_pref_df) berdasarkan dataframe utama (movie_df) dan melakukan seleksi yang mana data yang dimasukkan adalah movie yang termasuk kedalam top_movie_refference
5. Dan selanjutnya menghitung rata-rata rating yang diberikan oleh user

Gambar di bawah meupakan proses penerapan dari tahapan yang saya jelaskan di atas :
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/getUserMoviePreference.png?token=ANXJTPJSYPCC73TN5OKRV4DBSM7AU)


Gambar di bawah ini merupakan daftar 10 rekomendasi yang dihasilkan :
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/result.png?token=ANXJTPPCZ3JXL7HRSLGFUETBSM6PO)

## Evaluation
Untuk bagian Evaluasi, Saya menguji performa model ini dengan mean squared error (MSE), precision, dan recall. Menurut sumber yang saya temukan, kedua metrik ini sangat cocok untuk mengukur performa model machine learning. Berikut adalah penjelasan dari setiap metrik :

* [Mean Squared Error](https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html): Mean Squared Error (MSE) mungkin adalah fungsi loss yang paling sederhana dan paling umum, sering diajarkan dalam kursus pengantar Machine Learning. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi atau model peramalan seperti Moving Average, Weighted Moving Average dan Analisis Trendline. Cara menghitung Mean Squared Error (MSE) adalah melakukan pengurangan nilai data aktual dengan data peramalan dan hasilnya dikuadratkan (squared) kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. Nilai MSE yang didapatkan dari proyek ini adalah 0.0083
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/mse.png?token=ANXJTPLZDXNFX3JY5W5FX53BSM7GM)
Di bawah ini adalah grafik mse yang dihasilkan dari proses training model yang saya buat.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/MSE111.png?token=ANXJTPKLIKSF57F5NHUJ7JLBSM7GQ)

* [Precision](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Precision adalah tingkat ketepatan antara informasi yang diminta oleh pengguna dengan jawaban yang diberikan oleh sistem. Sedangkan recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Precision yang didapatkan dari proyek ini adalah 1.0000 ![image](https://www.mydatamodels.com/wp-content/uploads/2020/10/5.-Precision-formula.png)
Di bawah ini adalah grafik precision yang dihasilkan dari proses training model yang saya buat.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/Precision111.png?token=ANXJTPOQRZGLHL2DSWVNWKLBSM7LO)

* [Recall](https://dataq.wordpress.com/2013/06/16/perbedaan-precision-recall-accuracy/) : Recall adalah tingkat keberhasilan sistem dalam menemukan kembali sebuah informasi. Nilai Recall yang didapatkan dari proyek ini adalah 0.6907
Di bawah ini adalah grafik recall yang dihasilkan dari proses training model yang saya buat.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/Recall111.png?token=ANXJTPL4VA4WEH3BPWVSALLBSM7LS)

