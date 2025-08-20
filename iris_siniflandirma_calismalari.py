import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# Veri setini yükleme
df = pd.read_csv("iris.csv")

# Veri setinin kolonlarını ve ilk 5 satırını görüntüleme
print(df.columns)
print(df.head())

# "Id" sütununu kaldırma
df = df.drop("Id", axis=1)
print("id sütunu kaldırıldı.")
print(df.head())

# Veri setinde null değerleri kontrol etme
print("Null değerler:")     
print(df.isnull().sum())

# Veri setinin temel istatistiklerini görüntüleme
print("Veri setinin temel istatistikleri:")     
print(df.describe())

# Veri setinin bilgi özetini görüntüleme
print(df.info())

# Keşifsel Veri Analizi (Exploratory Data Analysis - EDA)
# pair plot ile veri setindeki ilişkileri görselleştirme
sns.pairplot(df, hue="Species", markers=["o", "s", "D"])
plt.show()

# Box Plot : Her bir özelliğin türlere göre dağılımını görmek için kutu grafikleri
plt.figure(figsize=(12, 6))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x="Species", y=feature, data=df)
plt.tight_layout()
plt.show()


X = df.drop("Species", axis=1)
y = df["Species"]

"""
# Veriyi eğitim ve test setlerine ayırma farklı bir yöntem kullanarak
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
"""
# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=15)

# Tunning yapmadan lojistik regresyon modeli oluşturma
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Modelin başarısını değerlendirme
print("Modelin başarı oranı:")
score = accuracy_score(y_pred, y_test)
print("score: ", score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n ", confusion_matrix(y_pred, y_test))

# Modelimiz mükemmel bir başarı yakaladı. Dolayısı ile hiperparametre tuning yapmaya gerek yok.
# Ancak çalışmada kodları eksik bırakmamak için GridSearchCV ile tuning yapacağız.
# Bu tamamen kodların gösterimi amaçlı. İeride tuning yaparken bu kodları kullanabilirsiniz
# Hiperparametre tuning için gerekli kütüphaneleri içe aktarma
# Tunning yaparak modelin başarısını artırma
from sklearn.model_selection import GridSearchCV, StratifiedKFold
penalty = ["l1", "l2", "elasticnet"]
c_values = [100, 10, 1, 0.1, 0.01]
solver = ["newton-cg", "lbfgs", "liblinear", "sag", "saga", "newton-cholesky"]

params = dict(penalty=penalty, C=c_values, solver=solver)

from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold()

grid = GridSearchCV(estimator = model, param_grid = params, cv=cv, scoring="accuracy")

import warnings
warnings.filterwarnings("ignore")

grid.fit(X_train, y_train)

print("En iyi parametreler: ", grid.best_params_)
print("En iyi skor: ", grid.best_score_)    

y_pred = grid.predict(X_test)

score = accuracy_score(y_pred, y_test)
print("score: ", score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n ", confusion_matrix(y_pred, y_test))
print("****************************************************************")

# Modelin başarı için başka adımlar
# one vs one yöntemi ile modelin başarısını artırma
# model mükemmel ancak kodlama öğrenimi açısından bu kodları da ekliyorum
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
onevsonemodel = OneVsOneClassifier(LogisticRegression())
onevsrestmodel = OneVsRestClassifier(LogisticRegression())

print("one vs one model için")
onevsonemodel.fit(X_train, y_train)
y_pred = onevsonemodel.predict(X_test)
score = accuracy_score(y_pred, y_test)
print("score: ", score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n ", confusion_matrix(y_pred, y_test))

print("****************************************************************")

print("one vs rest model için")
onevsrestmodel.fit(X_train, y_train)
y_pred = onevsrestmodel.predict(X_test)
score = accuracy_score(y_pred, y_test)
print("score: ", score)
print(classification_report(y_pred, y_test))
print("confusion matrix: \n ", confusion_matrix(y_pred, y_test))

print("****************************************************************")

# Iris veri setinde bu kez SVM ile sınıflandırma ile yapalım.
# Aynı test ve eğitim setlerini kullanacağız.
from sklearn.svm import SVC

# SVM modelini oluşturma ve eğitme
# Kernel olarak lineer kullanacağız
print("\nSVM linear modeli için")
svc=SVC(kernel='linear')
svc.fit(X_train,y_train)
#print("Katsayılar =",svc.coef_)
y_pred = svc.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# Kernel olarak rbf kullanacağız
print("\nSVM rbf modeli için")
rbf=SVC(kernel='rbf')
rbf.fit(X_train,y_train)
y_pred1=rbf.predict(X_test)
print(classification_report(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))

# SVM ile görselleştirme örneği aşağıdaki gibi yapılabilir.
# GÖRSELLEŞTİRME KODU

# 1. Adım: Görselleştirme için sadece 2 özellik seçelim (Petal Length ve Petal Width)
# Pairplot'tan bu ikisinin en iyi ayırımı sağladığını görmüştük.
X_vis = df[['PetalLengthCm', 'PetalWidthCm']]
y_vis = df['Species'] 

# Kategorik tür isimlerini (string) sayısallaştıralım (0, 1, 2 gibi). Bu, renklendirme için gereklidir.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_vis_encoded = le.fit_transform(y_vis)

# 2. Adım: Modelleri bu 2D veri ile yeniden eğitelim
svc_linear_2d = SVC(kernel='linear').fit(X_vis, y_vis_encoded)
svc_rbf_2d = SVC(kernel='rbf').fit(X_vis, y_vis_encoded)

# 3. Adım: Karar sınırlarını çizmek için bir fonksiyon oluşturalım
def plot_decision_boundaries(model, X, y, title):
    # Meshgrid (ızgara) oluşturma
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # 4. Adım: Izgaradaki her nokta için tahmin yapma
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 5. Adım: Arka planı renklendirme (karar bölgeleri)
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

    # 6. Adım: Gerçek veri noktalarını çizdirme
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    plt.xlabel('Petal Length (Cm)')
    plt.ylabel('Petal Width (Cm)')
    plt.title(title)
    # Efsane (legend) oluşturma
    handles, labels = scatter.legend_elements()
    plt.legend(handles=handles, labels=list(le.classes_), title="Species")
    plt.show()

# Fonksiyonu kullanarak görselleri oluşturalım
print("\nSVM (Linear Kernel) - 2D Karar Sınırları Görselleştirmesi")
plot_decision_boundaries(svc_linear_2d, X_vis, y_vis_encoded, 'SVM with Linear Kernel Decision Boundaries')

print("\nSVM (RBF Kernel) - 2D Karar Sınırları Görselleştirmesi")
plot_decision_boundaries(svc_rbf_2d, X_vis, y_vis_encoded, 'SVM with RBF Kernel Decision Boundaries')

# Bu kodlar, SVM ile Iris veri setinin 2D görselleştirmesini yapar.
# İki farklı kernel (linear ve rbf) ile karar sınırlarını görselleştirir.
# Her iki modelin karar sınırlarını ve veri noktalarını görsel olarak incelemenizi sağlar.
# Bu, modelin nasıl çalıştığını ve hangi türlerin nasıl ayrıldığını anlamanıza yardımcı olur.
# Ayrıca, modelin karar sınırlarının veri setindeki türleri nasıl ayırdığını görselleştirir.
# Bu tür görselleştirmeler, modelin performansını ve karar verme sürecini daha iyi anlamanızı sağlar.

# 3 boyutlu görselleştirme için 3D grafiğe geçelim
import plotly.express as px

# 3 Boyutlu interaktif bir saçılım grafiği (scatter plot) oluşturma
print("\nPlotly Express ile 3D İnteraktif Görselleştirme")

fig = px.scatter_3d(df, 
                    x='SepalLengthCm', 
                    y='PetalLengthCm', 
                    z='PetalWidthCm',
                    color='Species', # Noktaları türlere göre renklendir
                    title='Iris Veri Setinin 3 Boyutlu Görselleştirmesi',
                    labels={'SepalLengthCm': 'Sepal Uzunluğu (cm)', 
                            'PetalLengthCm': 'Petal Uzunluğu (cm)', 
                            'PetalWidthCm': 'Petal Genişliği (cm)'} # Eksen etiketlerini güzelleştir
                   )

# Grafiğin daha okunaklı olması için düzenlemeler
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40)) # Kenar boşluklarını azalt

# Grafiği göster
fig.show()

# Bu kod, Plotly Express kütüphanesini kullanarak Iris veri setinin 3 boyutlu interaktif bir görselleştirmesini yapar.
# Sepal uzunluğu, petal uzunluğu ve petal genişliği özelliklerini kullanarak
# her bir iris türünü farklı renklerle gösterir.
# Plotly Express, etkileşimli grafikler oluşturmak için harika bir araçtır
# ve bu tür görselleştirmeler, veri setindeki ilişkileri ve türler arasındaki farkları daha iyi anlamanızı sağlar.
# 3D görselleştirme, veri setindeki üç özellik arasındaki ilişkileri görselleştirmenin etkili bir yoludur
# ve türler arasındaki ayrımları daha net bir şekilde görmenizi sağlar.

# Farklı bir yöntem ile 4.boyutu ekleyelelim.
#Dördüncü bir uzamsal eksen ekleyemesek de, dördüncü özelliğimiz olan 
# SepalWidthCm'yi noktaların boyutunu (size) değiştirerek grafiğe dahil edebiliriz.
# Dördüncü boyutu (Sepal Width) noktaların boyutu olarak ekleyerek görselleştirme
fig2 = px.scatter_3d(df, 
                     x='SepalLengthCm', 
                     y='PetalLengthCm', 
                     z='PetalWidthCm',
                     color='Species',
                     size='SepalWidthCm',  # 4. boyut olarak Sepal Genişliğini ekle
                     hover_data=['Species'], # Fare ile üzerine gelince tür ismini göster
                     title='4D Iris Veri Görselleştirmesi (3D Scatter + Boyut)')

fig2.update_layout(margin=dict(l=0, r=0, b=0, t=40))
fig2.show()

# şimdi sıra Naive Bayes modelinde
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["Species"] = label_encoder.fit_transform(df["Species"])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred = gnb.predict(X_test_scaled)
print("\nNaive Bayes Modeli için")
print("confusion matrix: \n", confusion_matrix(y_pred, y_test))
print("accuracy score: ", accuracy_score(y_pred, y_test))
print("classification report: ", classification_report(y_pred, y_test))

# Naive Bayes modelinin görselleştirilmesi
# Karar Sınırlarını Görselleştirme (SVM ile Doğrudan Karşılaştırma İçin)
"""
Bu yöntem, Naive Bayes modelinin özellik uzayını hangi bölgelere ayırdığını gösterir. 
SVM için yazdığınız plot_decision_boundaries fonksiyonunu kullanarak bu modeli de kolayca görselleştirebiliriz. 
Bu, farklı modellerin aynı veriyi nasıl farklı "gördüğünü" anlamak için mükemmel bir yoldur.

Adımlar:

Görselleştirme için yine aynı 2 özelliği ('PetalLengthCm' ve 'PetalWidthCm') seçeceğiz.
Önemli: Modeli sadece bu 2D veri ile yeniden eğiteceğiz. Çünkü ana modeliniz 4 özellik üzerine kuruluydu.
Daha önce oluşturduğunuz görselleştirme fonksiyonunu bu yeni model ile çağıracağız.
"""

# GÖRSELLEŞTİRME KODU (Naive Bayes için)

# 1. Adım: Görselleştirme için 2 özellik seçelim
X_vis = df[['PetalLengthCm', 'PetalWidthCm']]
y_vis = df['Species'] # Henüz encode edilmemiş halini kullanalım, fonksiyon içinde encode ediliyor.

# Kategorik tür isimlerini sayısallaştıralım
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_vis_encoded = le.fit_transform(y_vis)

# 2. Adım: Gaussian Naive Bayes modelini bu 2D veri ile yeniden eğitelim
# Not: Burada ölçeklendirme yapmıyoruz çünkü Naive Bayes buna daha az duyarlıdır ve
# orijinal ölçekte görmek daha sezgisel olabilir.
gnb_2d = GaussianNB()
gnb_2d.fit(X_vis, y_vis_encoded)

# 3. Adım: SVM için yazdığınız karar sınırları çizme fonksiyonunu kullanalım
# Bu fonksiyonu kodunuzun üst kısımlarından tekrar çağırmanız yeterli.
# Eğer ayrı bir dosyadaysa, tekrar tanımlamanız gerekir.
def plot_decision_boundaries(model, X, y, title):
    # Meshgrid (ızgara) oluşturma
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Izgaradaki her nokta için tahmin yapma
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Arka planı renklendirme (karar bölgeleri)
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

    # Gerçek veri noktalarını çizdirme
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    plt.xlabel('Petal Length (Cm)')
    plt.ylabel('Petal Width (Cm)')
    plt.title(title)
    # Efsane (legend) oluşturma
    handles, labels = scatter.legend_elements()
    plt.legend(handles=handles, labels=list(le.classes_), title="Species")
    plt.show()

# Fonksiyonu Naive Bayes modeli için çağıralım
print("\nGaussian Naive Bayes - 2D Karar Sınırları Görselleştirmesi")
plot_decision_boundaries(gnb_2d, X_vis, y_vis_encoded, 'Gaussian Naive Bayes Decision Boundaries')
# Bu kod, Gaussian Naive Bayes modelinin karar sınırlarını görselleştirir.
# Özellikle SVM ile karşılaştırmak için kullanışlıdır.
# Naive Bayes, genellikle basit ve hızlı bir modeldir, ancak karar sınırları genellikle daha "yumuşak" ve daha az keskin olur.
# Bu tür görselleştirmeler, modelin nasıl çalıştığını ve hangi türlerin nasıl ayrıldığını anlamanıza yardımcı olur.
# Ayrıca, farklı modellerin karar sınırlarını karşılaştırarak hangi modelin veri seti için daha uygun olduğunu görselleştirir.

print("Tüm işlemler tamamlandı.")