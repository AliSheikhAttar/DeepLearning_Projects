#!/usr/bin/env python
# coding: utf-8

# ![0.jfif](https://uupload.ir/files/cy6_air.png)

# # Practice 6

# <div dir="auto">
# خسته نباشید =)
# 
# تا الان مطالب مربوط به CNN رو تموم کردیم و الان می تونیم شبکه های خیلی خفنی برای انجام کارای خیلی خفنی رو یاد گرفتیم و میتونیم کارای باحالی رو انجام بدیم

# <div dir="auto">
# در 2 تمرین آینده قراره که در مورد RNN ها صحبت کنیم.
# تسکی هم که میخوایم انجام بدیم NLP هستش.
# 
# 
# 

# # Word Embedding

# <div dir="auto">
# قبل از هر چیزی میخوایم برای هر کلمه یک embedding به دست بیاریم. برای این کار هم میشه embedding خومون رو ترین کنیم و هم  میشه از embedding های آماده استفاده کنیم

# <div dir="auto">
# حالا توی این تمرین، میخوایم که از embedding های معروف برای انجام یه کار ساده استفاده کنیم

# ![img](https://uupload.ir/files/ro4_1_uqw1pqumvzkm3geqtao5lq.png)

# # بخش اول

# <div dir="auto">
# سوال 1 :
# 
# چند مورد از embedding های معروف رو مورد بررسی قرار دهید.
# با سرچ کردن در اینترنت حداقل دو word embedding متفاوت پیدا کنید و در موردشون توضیح بدید که با استفاده از چه کلماتی ساخته شدن، همچنین لینک شون رو اینجا بذارید و در موردشون توضیح بدید
# 
# هر توضيحاتي که دوست داشتيد مي تونيد بنويسيد =)

# ## GloVe
# 

# GloVe یک (pre-trained embedding later) است که توسط تیم دانشگاه stanford آمریکا ارائه شده است که از روی اسم آن می توان به تعریفی از آن پی برد 
#  GloVe حاصل ترکیب Global و Vector است
#  الگوریتم یادگیری آن unsupervised learning است که شباهت و ارتباط معنایی کلمات با فاصله ی آن ها تعیین می شود
#  و به عنوان یک پروژه ی open-source در سال 2014 ارائه شد.
#  یادگیری این مدل توسط ماتریس ها برای یادگیری روابط خطی بین کلمات صورت می گیرد.
#  و در انتها در عمل بدست آمده که 
# 
# word2vec embedding layer Glove سریع تر است در مقایسه با 
# 
# time : word2vec embedding layer <  Glove
# کلماتی که بر روی آن ها train شده از چند منبع منحصر به فرد استخراج می شوند و میتوان از بین آن ها انتخاب کرد.
# 
# ---

# [Wikipedia 2014 + Gigaword 5 ](https://nlp.stanford.edu/data/glove.6B.zip)
# 
# [Twitter](https://nlp.stanford.edu/data/glove.twitter.27B.zip)

# ## Word2Vec

# word2vec  یک تکنیک و embedding layer از پیش آموزش داده شده است (pre-trained embedding layer) که توسط پژوهشگران google ارائه شده است که یک لیستی از اعداد یا به اصطلاح (vector) را شامل می شود که هر وکتور یک عدد را بیان می کند و با توجه به رابطه ی کسینوسی بین این وکتور ها می توان کلمه هایی که در یک خانواده ی معنایی قرار می گیرند مشخص شوند
# و دو نوع استفاده دارد: 
# 1- Skip-Gram
# این متد کلمات مرتبط با کلمه ی ورودی شبکه که از یک جمله انتخاب می شود را پیشبینی می کند
# 
# 2- (CBOW) = continuous bag of words
# یک جمله ی ناقص را با پیشبینی کردن کلمه یا کلمات مورد نظر کامل می کند که با پیدا کردن ارتباط با بررسی کلمات بکار رفته در جمله و چند کلمه قبل و بعد از کلمات مجهول پیشبینی خود را انجام می دهد و به این دلیل به این نام اسم گذاری شده چون ترتیب کلمات برایش مهم نیست
# 
# 
# شماتیک الگوریتم این دو متد به گونه ایست که گویا برعکس یکدیگر عمل می کنند 
# CBOW   کلمه ی مورد نظر را پیش بینی می کند با دریافت چند کلمه یا یک متن
# اما Skip-Gram با دریافت یک کلمه کلمه های مرتبط یا متن را پیش بینی می کند.
# 
# word2vec یادگیری خود را توسط 3 لایه ی شبکه ی عصبی انجام می دهد
# کلماتی که بر روی آن ها train شده نیز از چند منبع خاص استخراج می شوند و میتوان از بین آن ها انتخاب کرد.
# 
# ---

# [Google News 2013](http://vectors.nlpl.eu/repository/20/1.zip) 
# 
# 
# [English Wikipedia Dump of February 2017](http://vectors.nlpl.eu/repository/20/3.zip)

# # بخش دوم

# <div dir="auto">
# حالا وقتشه که از يکي از embedding هايي که بالاتر در موردشون صحبت کردين استفاده کنين

# <div dir="rtl">
# Embedding تون رو دانلود کنيد.
# براي دانلود کردن مي تونيد از چه دستوراتي استفاده کنيد؟ 
# <div dir="rtl">
# تفاوت دو دستور 
# <div dir="rtl">
# wget و axel
# <div dir="rtl">
#  در چه چيزي هست؟

#  wget از یک connection بیشتر برای دانلود استفاده نمی کند
# ولی در axel ما میتوانیم انتخاب کنیم که برای دانلود از چند connection استفاده کند و بجای دانلود فایل به طور کامل از ابتدا تا انتها آن را به بخش های مختلف تقسیم کرده و دانلود می کند و برای استفاده از آن ابتدا باید آن را نصب کرد.
# 
# 

# In[1]:


get_ipython().system('apt-get install axel')
from  IPython.display import clear_output
get_ipython().system('axel -n 9 http://nlp.stanford.edu/data/glove.6B.zip')
clear_output()


# <div dir="rtl">
# اگر فرمت اوليه ي فايلتون به صورت فشرده هست، اون رو آنفشرده کنيد =)
# 

# In[2]:


get_ipython().system('unzip -qx /content/glove.6B.zip')


# <div dir="rtl">
# 
# > Indented block
# 
# 
# چه فايل هايي داخل فايل فشرده تون قرار داره؟ در موردشون توضيح بدين.
# 
# به ازاي هرفايلي که داخل فايل فشرده تون هست، يه خط توضيح بنويسيد که در اصل چي هستش.
# 
# براي اضافه کردن هر توضيحات اضافه، روي دکمه ي 
# +Text 
# کليک کنيد

# 4 فایل تکست
# پسوند های *d که دارند نشان دهنده ی آن است که هر کلمه ای که دارد که در واقع همان embedding dimension است
# در داخل هر فایل تعداد زیادی کلمه به همراه تعداد زیادی از اعداد وجود دارد که در سوال بعد آن ها را توضیح می دهم.   

# <div dir="rtl">
# فايل اصلي تون به هر فرمتي هست رو نگاه کنيد. 
# ببينيد داخل فايل embedding اصلي تون چه اطلاعاتي هست
# 
# بعد اين که کدشو زديد، توضيحاتي در مورد اطلاعات بنويسيد

# In[3]:


get_ipython().system('head -6 /content/glove.6B.200d.txt')


# In[4]:


# !cat /content/glove.6B.200d.txt


# شامل کلمات و اعداد منحصر به آنها است
# 
# در هر خط با توجه به که در اینجا 200 است ، عدد وجود دارد که منحصر به کلمه ی ابتدای خط هستند که به نوعی فیچر های آن محسوب می شوند. (embedding dimension)
# 
# ---
# 
# 

# ## بخش سوم

# <div dir="rtl">
# حالا بايد يک ديکشنري بسازيد که کليد هاش کلمه هاي شما باشه و value 
# هاش embedding اون کلمه باشه

# In[5]:


word2embedding = {}
import numpy as np
f = open('/content/glove.6B.200d.txt')
for line in f:
  words = line.split()
  key = words[0]
  values = np.array(words[1:], dtype = 'float32')
  word2embedding[key] = values


# In[6]:


word2embedding['god']


# <div dir="rtl">
# Embedding 
# يک کلمه ي دلخواه رو نگاه کنيد ببينيد که چجوريه

# In[7]:


word2embedding['italy']


# <div dir="rtl">
# حالا ميخوايم 3 کلمه در نظر بگيريم. کلمه ي اول و دوم يه ارتباطي با هم دارن، م ميخواهيم يه کلمه ي چهارم پيدا کنيم که ارتباطش با کلمه ي سوم مثل ارتباط کلمه ي دوم با اول

# <div dir="rtl">
# مثال 
# 
# کلمه ي اول : ايران
# 
# کلمه ي دوم : تهران
# 
# کلمه ي سوم آلمان
# 
# کلمه ي چهارم(که قراره به دست بياد) : برلين

# <div dir="rtl">
# براي اين کار بالا، يک تابع بنويسيد که 3 کلمه رو و ديکشنري embedding رو بگيره بگيره و کلمه ي چهارم رو پيدا کنه و اون رو پرينت کنه

# <div dir="rtl">
# سوال مهم : با چه معياري اين کار رو ميخوايد اينجام بديد؟ 
# 
# براي مثال توي کلاس از معيار فاصله اقليدسي استفاده کرديم. يک روش ديگه استفاده از شباهت به جاي فاصله ميتونه باشه.
# 
# براي مثال cosine similarity
# يک روش از اين دو تا رو به صورت دلخواه انتخاب کنيد و کلمه ي چهارمتون رو پيدا کنيد =)

# In[8]:


from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np


# ## Cosine similarity

# In[21]:


def find_connection(word1, word2, word3, embedding_dict):
  words = []
  distances_list = []
  vectorI   = embedding_dict[word1]
  vectorII  = embedding_dict[word2]
  vectorIII = embedding_dict[word3]
  expected = cosine_distances(vectorI.reshape(1,-1), vectorII.reshape(1,-1))
  for key, value in embedding_dict.items():
    result = cosine_distances(vectorIII.reshape(1,-1), value.reshape(1,-1))
    vector_distance = np.linalg.norm(expected - result)
    distances_list.append(vector_distance)
    words.append(key)
  distances_args = np.argsort(distances_list)
  for i in range(5):
    print(words[distances_args[i]])
  print(f"\n the most related one is {words[distances_args[0]]}.")

find_connection('iran', 'tehran', 'germany', word2embedding)


# ## Euclidean distance

# In[19]:


def find_connection(word1, word2, word3, embedding_dict):
  words = []
  distances_list = []
  vectorI   = embedding_dict[word1]
  vectorII  = embedding_dict[word2]
  vectorIII = embedding_dict[word3]
  expected = (vectorI - vectorII)
  for key, value in embedding_dict.items():
    result = (vectorIII - value)
    vector_distance = np.linalg.norm(expected - result)
    distances_list.append(vector_distance)
    words.append(key)
  distances_args = np.argsort(distances_list)
  for i in range(3):
    if words[distances_args[i]] != 'germany' and words[distances_args[i]] !='german': #بدیهی است
      break
    else: 
      i +=1
  print(words[distances_args[i]])

find_connection('iran', 'tehran', 'germany', word2embedding)


# <div dir="rtl">
# موفق باشيد 

# <div dir="rtl">
# 

# <div dir="rtl">
# 
