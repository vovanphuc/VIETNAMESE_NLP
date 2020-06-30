# VIETNAMESE_NLP
# Trình bày 4 trong các bài toán đặc trưng của xử lý ngôn ngữ tự nhiên Tiếng Việt :
1. So sánh 2 thuật toán machine learning (KNN, SVM) và 1 mô hình CNN đơn giản trong nhận diện chữ số viết tay
2. Nhận diện chữ viết tay tiếng việt theo từng dòng ( cụ thể là chữ viết tay địa chỉ tiếng Việt )
3. Phân loại văn bản tiếng Việt
4. Sửa lỗi chính tả tiếng Việt

# Dữ liệu của từng bài toán :
1. Tập dữ liệu MNIST
2. Tập dữ liệu từ cuộc thi Cinnamon AI Marathon 2018
3. 10000 văn bản tiếng Việt theo 10 chủ đề
4. 10000 văn bản tiếng Việt đúng chính tả

# Model của từng bài toán :
1. KNN, SVM, CNN
2. CRNN + CTCLoss
3. LSTM
4. Bidirectional LSTM + Attention

# So sánh độ chính xác của 2 thuật toán machine learning (KNN, SVM) và 1 mô hình CNN đơn giản trong bài toán nhận diện chữ số viết tay

![](https://github.com/vovanphuc/VIETNAMESE_NLP/blob/master/img/danhgia1.png)

# Kết quả đạt được của từng bài toán :

1. Nhận diện chữ số viết tay:

![](https://github.com/vovanphuc/VIETNAMESE_NLP/blob/master/img/ketqua1.png)

2. Nhận diện chữ viết tay là địa chỉ :

![](https://github.com/vovanphuc/VIETNAMESE_NLP/blob/master/img/ketqua2.png)

3. Phân loại văn bản :

![](https://github.com/vovanphuc/VIETNAMESE_NLP/blob/master/img/ketqua4.png)

4. Sửa lỗi chính tả : 

![](https://github.com/vovanphuc/VIETNAMESE_NLP/blob/master/img/ketqua3.png)

# Liên Hệ :
Mình giới hạn dung lượng của github nên mình không thể upload toàn bộ các model của bài toán, các bạn cần có thể liên hệ mình :
Facebook : https://www.facebook.com/VoVanPhucc
Skype : vovovanphuc100598

# Tài liệu tham khảo :
1. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
2. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
3. https://towardsdatascience.com/an-introduction-to-attention-transformers-and-bert-part-1-da0e838c7cda
4. https://towardsdatascience.com/sentence-classification-using-bi-lstm-b74151ffa565
5. https://github.com/githubharald/CTCWordBeamSearch
