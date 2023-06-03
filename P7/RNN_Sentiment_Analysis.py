#!/usr/bin/env python
# coding: utf-8

# ![0.jfif](https://uupload.ir/files/cy6_air.png)
# 
# 
# 
# 

# # Practice 7

# <div dir="rtl">
# سلام!
# 
# در این تسک باید روی یک دیتاست، عمل شناسایی احساسات رو انجام بدین.

# ## Kaggle Token and downloading data

# <div dir="rtl">
# در این قسمت ابتدا باید دیتای مورد نظر رو از سایت کگل دانلود کنید، پیشنهاد ما استفاده از دیتاست IMDB هستش که شامل نظرات در مورد فیلم های مختلف می باشد

# <div dir="rtl">
# 
# این دیتاست رو می تونید از لینک زیر پیدا کنید
# 
# https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# <div dir="rtl">
# 
# وارد لینک بالا بشید و اطلاعات گفته شده در مورد دیتاست رو مطالعه کنید

# <div dir="rtl">
# 
# حال ابتدا توکن kaggle api خود را آپلود کنید.
# 

# In[42]:


from IPython.display import clear_output
get_ipython().system('pip install --upgrade kaggle')
clear_output()
from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
get_ipython().system('mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json')


# <div dir="rtl">
# 
# حالا دیتاست تون رو دانلود بفرمایید

# In[43]:


get_ipython().system('kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews')


# <div dir="rtl">
# 
# فایلی که دانلود کردید به صورت زیپ می باشد پس باید اون رو اکسترکت

# In[44]:


get_ipython().system('unzip -qx /content/imdb-dataset-of-50k-movie-reviews.zip')


# <div dir="rtl">
# 
# خب حالا وقتشه با توجه به کارایی که میخواید انجام بدید یه تعدادی از کتابخونه های مورد نیازتون رو ایمپورت کنید.
# برای مثال فایلی که داخل فایل فشرده بود دارای فرمت csv هستش پس میتونید از پانداز استفاده کنید!

# In[45]:


import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# <div dir="rtl">
# 
# فایل csv رو بخونید

# In[46]:


df = pd.read_csv("/content/IMDB Dataset.csv")


# <div dir="rtl">
# 
# یه نکاه اجمالی بندازید که دیتافیمتون به چه صورت هستش

# In[47]:


df.head()


# <div dir="rtl">
# 
# همون طور که می بینید، دیتافریم شما دارای 2 ستون نظر و  لیبل اون احساس هستش
# 
# قبل از هر چیز، می خوایم که دو تا کار انجام بدیم

# <div dir="rtl">
# 
# 1. ببینیم که تعداد کلمات هر نظر چه تعداد هستش برای این کار میتونید از متد apply برای دیتافریم استفاده کنید.
# و یک ستون جدید به دیتافریم اضافه کنید که تعداد کلمات هر نظر داخلش باشه.
#  اسم ستون رو length بذارید
# 
# پ.ن : متد apply صرفا پیشنهاد بود، می تونید از روش های دیگه هم استفاده کنید

# In[48]:


def numberofwords(a):
  b = a.split()
  return len(b)
number_apply = df.apply(lambda row: numberofwords(row['review']),axis=1)
df['length'] = number_apply


# In[49]:


df.head()


# <div dir="rtl">
# 
# حالا که ستون جدید رو ایجاد کردید، میخوایم بدونیم که میانگین تعداد کلما هر جمله چندتاست، می تونید از mean استفاده کنید ولی خب دستور describe باحال تره. یادتون نره که این کارو میخواید روی ستون جدیدتون انجام بدید

# In[50]:


df['length'].describe()


# <div dir="rtl">
# 
# همچنین، همون طور که میبینید، ستون sentiment که قراره لیبلتون باشه، دارای مقادیر متنی هستش که بایستی به عددی تبدیلشون کنید.
# 
# برای این کار می تونید از دستور get_dummies استفاده کنید.
# 
# اگه نحوه ی کارشو بلد نیستید با یه سرچ ساده می تونید پیداش کنید

# In[51]:


df = pd.get_dummies(df, columns=['sentiment'], drop_first=True, prefix="", prefix_sep="")


# <div dir="rtl">
# 
#  تا اینجای کار دیتافریم شما باید 3 تا ستون داشته باشه که تو یکیش متن های شما هستش، یکی هم شامل طول هر کدوم از جملات و اون یکی هم شامل لیبل هر نظر هستش

# In[52]:


df = df.rename(columns={'positive':'sentiment'})


# <div dir="rtl">
# 
# حالا باید تمامی جملات رو در یک لیست بریزید، همچنین تمامی لیبل ها نیز داخل یک لیست دیگر قرار گیرند

# In[53]:


sentences = list(df['review'])
labels = list(df['sentiment'])
samples_size = len(df)


# <div dir="rtl">
# 
# در ادامه باید پارامترهای مورد نظر برای حداکثر تعداد کلمات موجود در جمله، تعداد کلمات موجود در دیکشنری (برای قرار گرفتن در امبدینگ) و توکن کلمات خارج از دیکشنری را تعریف کنید

# <div dir="auto">
# 
# حداکثر طول جملات را با توجه به اطلاعات بالا خودتون تنظیم کنید!

# In[54]:


vocab_size = 50000
max_length = df['length'].max()
oov_token = "<OOV>"


# <div dir="rtl">
# 
# سپس بایستی Tokenizer تون رو بسازید و سپس اون رو روی جملاتتون فیت کنید

# In[55]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token= oov_token)
tokenizer.fit_on_texts(sentences)


# <div dir="rtl">
# 
# اکنون جملات تون رو با استفاده از Tokenizer تون به سری هایی از اعداد صحیح تبدیل کنید و اسم اون رو sequences بذارید

# In[56]:


sequences = tokenizer.texts_to_sequences(sentences)


# <div dir="rtl">
# 
# الان هم وقت اونه که همه ی جملاتتون رو هم طول کنید، با استفاده از توابع کراس این کار رو انجام  بدید و طول اون سری هایی رو که ساختید به یک عدد خاص برسونید (اون عدد خاص چنده؟ راهنمایی می کنم، لزومی نداره 32 باشه مثل دفعه ی پیش و بالاتر یه کارایی کردید که راحت تر بتونید این عدد رو انتخاب کنید)

# max_length

# In[57]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(sequences, maxlen = max_length, padding = 'post', truncating= 'post')


# <div dir="rtl">
# 
# نوبتی هم اگر باشه نوبت تقسیم داده ها به ترین و تست هستش، برای این کار چجوری میخواید عمل کنید؟
# 
# آفرین! از تابع ترین تست اسپلیت استفاده کنید =)
# 
# یادتون باشه که رندوم استیت رو برابر 101 قرار بدید!

# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size = 0.2, random_state = 101)
y_train = np.array(y_train)
y_test = np.array(y_test)


# <div dir="rtl">
# 
# الانم وقتشه که شکل دیتاهاتون رو ببینید

# In[59]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# <div dir="rtl">
# 
# مثل همیشه مدلتون رو بسازید!
# 
# 1. بعد لایه امبدینگ تون رو مشخص کنید که چند تا باشه، یادتون باشه که هرچی بیشتر باشه مدلتون سنگین تره
# 
# 2. یک لایه ی LSTM کافی هستش و تعداد نود هاش رو برابر 64 بذارید.
# اگه دوست داشتید میتونید بیشتر از یک لایه استفاده کنید که تاثیرشو ببینید ولی اون موقع باید پارامتر return_sequences رو توی LSTM اولی برابر True قرار بدید
# 
# 3. مثل همیشه یه لایه ی دنس، تعداد نود هاش رو هم برابر 64 بذارید و اکتیوشن relu استفاده کنید
# 
# 4. لایه ی خروجی یادتون نره!

# In[60]:


embedding_dimension = 150
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dimension ,input_length= max_length))
model.add(LSTM(64))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# <div dir="rtl">
# 
# و صد البته که باید مدلتون رو کامپایل کنید!

# In[61]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# <div dir="rtl">
# 
# حالا هم مدلتون رو روی دیتاتون فیت کنید

# In[62]:


num_epochs = 10
history = model.fit(X_train, y_train,
                    epochs = num_epochs, validation_data = (X_test,y_test),
                    verbose = 1, batch_size = 128)


# <div dir="rtl">
# 
# موفق باشید =)
# 
