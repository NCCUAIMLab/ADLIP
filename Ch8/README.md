# Ch8 手寫辨識與影像還原 專案說明
MNIST 是含有手寫數字的資料集，有 60,000 張訓練資料以及 10,000 張測試資料
請透過連結 (http://yann.lecun.com/exdb/mnist/) 下載 MNIST 資料集

2.1 請建構一個分類網路來分類 MNIST 的數字，且此網路基於多層卷積網路建構而成(至少五層)，並查看此時網路預測的準確度。(提示：可以使用交叉熵(Cross Entropy)當作損失函數)

2.2 請隨機將 5%, 10%, 15% 的像素值設為 255 作為干擾的雜訊，並使用您在 2.1 訓練完成的模型來分類這些受到干擾的影像，並查看在這三種干擾下的準確度。

2.3 請重新訓練您在 2.1 建構出的網路，此時的訓練集改集使用受過雜訊干擾的影像，並同樣測試在這三種雜訊干擾的影像，查看其準確度。

2.4 承 2.2, 請建構一個還原網路(同樣是卷積神經網路)，此網路的輸入是一張受雜訊干擾的影像，網路的輸出應是一張乾淨影像。(提示：損失函數可以使用均方誤差(MSE)或是平均絕對值誤差(MAE))

2.5 請使用您建構的還原網路，還原不同程度的受損影像，並使用您在 2.1 訓練好的網路預測結果，查看其準確度。

## 英文說明
The MNIST dataset contains handwritten digits, with a training set of 60,000 samples and a test set of 10,000 samples.
Please download it here: http://yann.lecun.com/exdb/mnist/

2.1 (20%) You are asked to construct a classification model based on multi-layer convolutional neural networks (at least five layers) for digit recognition. Please report the prediction accuracy for the test set. (Hint: its loss function could be cross entropy)

2.2 (15%) Please randomly set 5%, 10%, and 15% of pixels to 255 for each image and evaluate the test set using the model you trained on clean images from 2.1. Report the prediction accuracies for the three different corruption rates. Compare your results with those in Question 2.1. What do you find?

2.3 (15%) Following Question 2.2, please re-train your model with the corrupted data (5%, 10%, and 15% separately) and re-evaluate the test set.
Report the prediction accuracies also for the three different corruption rates. Compare your results with those from 2.1 and 2.2. What do you find?
  
2.4 (15%) Following Question 2.2, please construct ONE restoration model (also convolutional neural networks) that inputs a corrupt image and outputs its restored image. (Hint: its loss function could be MAE or MSE)

2.5 (15%) Following Question 2.4, please evaluate the restored test set using the model you trained on clean images in Question 2.1. Report the prediction accuracies for restored images separately at the three different corruption rates. What do you find?