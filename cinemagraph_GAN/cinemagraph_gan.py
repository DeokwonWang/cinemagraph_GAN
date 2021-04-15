from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image

# 폴더 생성
import os
if not os.path.exists("./gan_images/3"):
    os.makedirs("./gan_images/3")

# 무작위 랜덤시드 설정(아무숫자나 상관없음)
np.random.seed(3)
tf.random.set_seed(3)

# 생성자 모델
generator = Sequential()
generator.add(Dense(128*32*32, input_dim=100, activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())
generator.add(Reshape((32, 32, 128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=5, padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(3, kernel_size=5, padding='same', activation='tanh'))

# 판별자 모델
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(128,128,3), padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False

# 생성자와 판별자 모델을 연결시키는 gan 모델 생성
ginput = Input(shape=(100,))
dis_output = discriminator(generator(ginput))
gan = Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
gan.summary()

# 데이터 불러오기
X = np.load("./cinemagraphnumpy.npy")
print(X.shape)

# 신경망을 실행시키는 함수
def gan_train(epoch, batch_size, saving_interval):

  # 데이터 불러오기
  X_train = X
  X_train = X_train.reshape(X_train.shape[0], 128, 128, 3).astype('float32')

  # 픽셀값은 0에서 255사이, 127.5를 빼준 뒤 127.5로 나누어 줌으로 -1에서 1사이의 값으로 바꿈
  X_train = (X_train - 127.5) / 127.5  

  true = np.ones((batch_size, 1))
  fake = np.zeros((batch_size, 1))

  for i in range(epoch):
          # 실제 데이터를 판별자에 입력
          idx = np.random.randint(0, X_train.shape[0], batch_size)
          imgs = X_train[idx]
          d_loss_real = discriminator.train_on_batch(imgs, true)

          # 가상 이미지를 판별자에 입력
          noise = np.random.normal(0, 1, (batch_size, 100))
          gen_imgs = generator.predict(noise)
          d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

          # 판별자와 생성자의 오차를 계산
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
          g_loss = gan.train_on_batch(noise, true)

          # 진행상황 출력(에포크,d_loss,g_loss)
          print('epoch:%d' % i, ' d_loss:%.4f' % d_loss, ' g_loss:%.4f' % g_loss)

        # 중간 과정을 이미지로 저장
        # 만들어진 이미지들은 gan_images 폴더에 저장
          if i % saving_interval == 0:
              
              noise = np.random.normal(0, 1, (batch_size, 100))
              gen_imgs = generator.predict(noise)

              # 이미지 리스케일링 0 - 1
              gen_imgs = 0.5 * gen_imgs + 0.5

              # 컬러 이미지로 전환
              color_imgs = gen_imgs[0, :, :, :] * 255
              color_imgs = color_imgs.astype(np.uint8)
              img = Image.fromarray(color_imgs)    
              
              # 이미지 저장
              img.save("gan_images/3/gan_%.4d.png" % i)

gan_train(4001, 10, 50)  # 4000번 반복되고(+1), 배치 사이즈는 10, 50번 마다 결과가 저장

# 학습된 모델 저장
from keras.models import load_model
gan.save('gan_model_003.h5')