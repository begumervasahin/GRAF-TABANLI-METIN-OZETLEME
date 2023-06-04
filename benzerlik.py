from ctypes.wintypes import RGB
from tkinter import Frame, Tk, Button, Label, Entry
from tkinter.filedialog import askopenfilename
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from rouge import Rouge
import math
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
import numpy as np
from gensim.models import Word2Vec 
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def select_file():
    # Dosya seçme iletişim kutusunu aç
    filename = askopenfilename()
    # Seçilen dosya yolunu etikete yazdır
    label.config(text=filename)
    # Dosya içeriğini oku
    with open(filename, 'r') as file:
        contents= file.read()
 

        # Cümleleri parçala ve listeye ekle
        sentences = contents.split('.')
        print(sentences)
        
        # Kullanıcıların girdiği sayıları al
        num_keywords = float(keyword_entry1.get())
        num_sentences = float(sentence_entry.get())
        print(num_keywords, num_sentences)

        # Boş bir graf oluştur
        G = nx.Graph()
        # Cümleleri grafın düğümleri olarak ekle
        for sentence in sentences:
            G.add_node(sentence.strip())
        # Kelimeler arasında bağlantılar kur
        for i in range(len(sentences)-1):
            G.add_edge(sentences[i].strip(), sentences[i+1].strip())
        # Grafiği çiz
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        # Her düğümün etiketi olarak cümleyi göster
        labels = {}
        for node in G.nodes():
            labels[node] = node
        nx.draw_networkx_labels(G, pos, labels)
        plt.show()

    cumleler = nltk.sent_tokenize(contents)
    # özel isimleri bul, p1 parametresi burada yapıldı///////////////////////////////////////////////////////////////////////////////////////      (P1)
    ozel_isim_sayisi = 0
    for cumle in cumleler:
        words = nltk.word_tokenize(cumle)
        tagged = nltk.pos_tag(words)
        for word, tag in tagged:
            if tag == 'NNP':
                ozel_isim_sayisi += 1
                #print(word)
            
    if ozel_isim_sayisi != 0:
        ozel_isimlerin_orani = len(cumleler) / ozel_isim_sayisi
        print("cumledeki özel isim oranı",ozel_isimlerin_orani)
    else:
        print("Metinde özel isim bulunamadı.")

    #nümerik veri  bul, p2 parametresi burada yapıldı///////////////////////////////////////////////////////////////////////////////////////      (P2)
    Cumleler = contents.split('.')
    for sentence in cumleler:
        words = nltk.word_tokenize(sentence)
        num_count = 0
        for word in words:
            if word.isdigit():
                num_count += 1
        if num_count !=0:
            numerik=num_count/len(words)
            print("Cümledeki numerik veri oranı ", numerik)

    #Cümle benzerliği threshold’unu geçen node’ların bulunması, p3 parametresi burada yapıldı//////////////////////////////////////////////      (P3)
    #burda if in kontrolü benzerlik oranından büyük veya eşit ise olmalı 
    if num_keywords!=0:
        threshold = (num_keywords/len(cumleler))
        print("Cümle benzerliği threshold’u ", threshold)
  
    #Cümlede başlıktaki kelimelerin olup olmadığının kontrolü (P4)  parametresi burada yapıldı//////////////////////////////////////////////    (P4)
    # başlıktaki kelimeleri belirle
    title = contents.split('.')[0]
    title_words = nltk.word_tokenize(title)

    # cümleleri parçala ve listeye ekle
    sentences = nltk.sent_tokenize(contents)
    baslik = []
    # her bir cümle için başlık kelime sayısını hesapla
    for sentence in sentences:
        kelime = nltk.word_tokenize(sentence)
        a1 = sum(1 for w in kelime if w in title_words)

        if a1 !=0:
          
          son= a1 / len(words)
          baslik.append(son)
         # tüm cümlelerin temel kelime benzerlik oranlarının ortalaması
    if len(baslik) != 0:
      baslik_orani= sum(baslik) / len(baslik)
      print("Tüm cümlelerin başlığa oranı: {:.2f}".format(baslik_orani))
  


    #Her kelimenin TF-IDF değerinin hesaplanması, p5 parametresi burada yapıldı//////////////////////////////////////////////      (P5)
    words = nltk.word_tokenize(contents)
    word_count = len(words)
    # kelime frekansları hesaplanıyor
    word_freq = Counter(words)
    # dokümandaki kelime sayısının %10'u hesaplanıyor
    num_keywords = int(word_count * 0.1)
    # en sık geçen num_keywords kelime belirleniyor
    keywords = [w for w, _ in word_freq.most_common(num_keywords)]
    # cümleleri parçala ve listeye ekle
    sentences = nltk.sent_tokenize(contents)

    # her bir cümle için temel kelime sayısı hesapla
    ratios = []
    for sentence in sentences:
       
        words = nltk.word_tokenize(sentence)
        num_keywords_in_sentence = sum(1 for w in words if w in keywords)
        if num_keywords_in_sentence !=0:
         ratio = num_keywords_in_sentence / len(words)
         ratios.append(ratio)
    # tüm cümlelerin temel kelime benzerlik oranlarının ortalaması
    if len(ratios) != 0:
        mean_ratio = sum(ratios) / len(ratios)
        print("Tüm cümlelerin TF-IDF değerini oranı: {:.2f}".format(mean_ratio))
    else:
        print("Dokümanda temel kelime yok.")

    
    #Cümle Skoru Hesaplama Algoritmasının Geliştirilmesi
    Cümle_Skor = (ozel_isimlerin_orani + numerik + threshold + baslik_orani + mean_ratio) / 5
    print("Cümle skoru: {:.2f}".format(Cümle_Skor))

   
    stemmer = PorterStemmer()
    # stop words listesini yükleyin
    stop_words = set(stopwords.words('english'))
    model = api.load("glove-wiki-gigaword-50")
#burada kendi oluşturduğumuz Cümle skorunu kullanıo oluşturduğumuz özetmee algoritması 
    kelimelersay = {}
    cumlelersay = {}
    tekrar =[]
    ozet =""
    # kelimeleri " ", cümleleri . lardan böldüm
    kelimeleri = contents.split(" ")
    cumleleri = contents.split(".")
    
# kelimelerin toplamda kaç tane geçtiğini sayıyor. Her kelimeye tekrarına göre puan veriyor.
    for kelime in kelimeleri:
        if kelimelersay.get(kelime) is not None:
            kelimelersay[kelime] +=1
        else:
            kelimelersay[kelime] =1



    # 2 den fazla geçen kelimeleri alıyor. Testlerde 2 en doğru sonucu verdi değişebilir.
    for kelime in kelimelersay:
         if kelimelersay.get(kelime) > 3:
            tekrar.append(kelime)

     # kelimelerin puanları ile cümleleri puanlandırıyor. Cümlelerin puanlarını elde ediyor.
    for cumle in cumleleri:
         for kelime in tekrar:
               if cumle.find(kelime) > -1:
                 if cumlelersay.get(cumleleri.index(cumle)) is not None:
                       cumlelersay[cumleleri.index(cumle)] += 1
                 else:
                     cumlelersay[cumleleri.index(cumle)] = 1

    print("-------------------------------")
    # puanı belirli sayıdan fazla olan cümleleri yazıyor.
    for a in cumlelersay:
           if cumlelersay[a] > 3: # Bu kısım yani Özetleme Düzeyi Metin Boyutu ve İçeriğine göre değiştirilecek.
             ozet += cumleleri[int(a)]
    print(ozet)
    print("-------------------------------")
    print("Metin Uzunluğu (Karakter): {}".format(len(contents)))
    print("Özet Uzunluğu (Karakter): {}".format(len(ozet)))
    
    def rouges():
    #ROUGE SKORUNU HESAPLADIK.
        rouge = Rouge()
        scores = rouge.get_scores(ozet, contents)
        #return scores[0]  # İlk skoru döndürmek için
        print("\n\n\n")
        
        print("Rouge score: " + str(scores))
        print("\n\n\n")



    # Özetlemeyi ayrı bir dosyaya yazdırmak için:
    summary_file_path = "ozet.txt"
    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write(ozet)





     # Özetlemeyi ayrı bir dosyaya yazdırmak için:
    summary_file_path = "Bizim_ozet.txt"
    with open(summary_file_path, "w", encoding="utf-8") as file:
        file.write(ozet)

 
    
    def wordEmbedding(fileName2,fileName3,fileName4):#WORD-EMBEDDİNG VE KOSİNÜS BENZERLİĞİNİ HESAPLADIK

        with open(fileName2, "r") as file:#ÖNCELİKLE VERİLEN VERİ DOSYASI OKUNUR
            dosya2 = open(fileName3,"w")#WORD-EMBADDING ÇIKTISI YAZILACAK
            dosya3 = open(fileName4,"w")#KOSİNÜS BENZERLİĞİ YAZILACAK

            sentence_embeddings = []#KOSİNÜS BENZERLİĞİ İÇİN

            for cumle in cumleler:
                    
                        #CÜMLELERİN BENZERLİK ALGORİTMASI:
                        cumle = cumle.translate(str.maketrans('', '', string.punctuation))# NOKTALAMA İŞARETLERİNİ KALDIRDIK
                        kelimeler = cumle.split()#KELİMELERE AYIRDIK
                        filtered_kelimeler = [kelime for kelime in kelimeler if not kelime.lower() in stop_words]#STOP-WORDLERİ ÇIKARDIK
                        stemmed_kelimeler = [stemmer.stem(kelime) for kelime in filtered_kelimeler]#STEAMING UYGULADIK(KELİME KÖKÜ BULMA)

                        # KELİME GÖMME UYGULADIK

                        sentence_embedding = []#WORD-EMBADDING İÇİN
                    
                        for kelime in stemmed_kelimeler:
                            try:
                                word_embedding = model[kelime]
                                sentence_embedding.append(word_embedding)#MODELLE KARŞILAŞTIRIP SONUCU LİSTEYE YAZDIK
                            except KeyError:
                                continue
                            
                        # CÜMLE VEKTÖRÜNÜ HESAPLADIK
                        if len(sentence_embedding) > 0:
                            sentence_embedding = np.mean(sentence_embedding, axis=0)
                            sentence_embeddings.append(sentence_embedding)#LİSTEYE YAZDIK
                            dosya2.write(" ".join(str(x) for x in sentence_embedding) + "\n")

                            #print("cümle vektörü:")
                            #print(sentence_embedding)

            sentence_embeddings = np.array(sentence_embeddings)
            kosinüs_matrix = cosine_similarity(sentence_embeddings)#KOSİNÜS BENZERLİĞİNİ HESAPLADIK
            print("\nkosinüs benzerliği\n")
            print(kosinüs_matrix)


            dosya3.write(" ".join(str(x) for x in kosinüs_matrix) + "\n\n")



            #KOSİNÜS MATRİSİNDEN TEK DEĞER DÖNDÜRME:
            ortalama = np.mean(kosinüs_matrix)
            print("Matrisin elemanlarının ortalaması:", ortalama)

   
                
    wordEmbedding("C:/Users/ervas/OneDrive/Masaüstü/yazlab3/d.txt","C:/Users/ervas/OneDrive/Masaüstü/yazlab3/cumle_vektoru.txt","C:/Users/ervas/OneDrive/Masaüstü/yazlab3/kosinus_benzerligi.txt")
   # wordEmbedding("C:/Users/sedan/Desktop/verilen.txt","C:/Users/sedan/Desktop/cumle_vektoru.txt","C:/Users/sedan/Desktop/kosinus_benzerligi.txt")
    rouges()




# Tkinter penceresi oluşturma
window = Tk()
window.title("Seva")

# Pencere boyutunu ve konumunu belirle
window_width = 500
window_height = 500
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Butonları ortalamak için satırlar oluştur
button_row = Frame(window, pady=10,bg='#EBDEF0')
button_row.pack()

keyword_row = Frame(window, pady=10,bg='#EBDEF0')
keyword_row.pack()

sentence_row = Frame(window, pady=10,bg='#EBDEF0') 
sentence_row.pack()

# Butonları satırlara yerleştir ve ortala
button = Button(button_row, text="Dosya Seç", command=select_file, bg="#C39BD3", width=30, height=2)
button.grid(row=0, column=0, sticky="nsew")

# Seçilen dosyanın yolunu gösteren etiket
label = Label(window, text="")
label.pack()

# Anahtar kelime sayısı için giriş kutusu
keyword_label1 = Label(keyword_row, text="Cümle benzerliği threshold :",bg="#C39BD3", width=30, height=1)
keyword_label1.grid(row=0, column=0, sticky="nsew")
keyword_entry1 = Entry(keyword_row,bg="#EAECEE")
keyword_entry1.grid(row=0, column=1, sticky="nsew")

# Cümle sayısı için giriş kutusu
sentence_label = Label(sentence_row, text="Cümle Skor threshold :",bg="#C39BD3", width=30, height=1)
sentence_label.grid(row=0, column=0, sticky="nsew")
sentence_entry = Entry(sentence_row,bg="#EAECEE")
sentence_entry.grid(row=0, column=1, sticky="nsew")

# Tüm satırların eşit genişlikte olduğundan emin olmak için columnconfigure kullan
window.columnconfigure(0, weight=1)
button_row.columnconfigure(0, weight=1)
keyword_row.columnconfigure(0, weight=1)
keyword_row.columnconfigure(1, weight=1)
sentence_row.columnconfigure(0, weight=1)
sentence_row.columnconfigure(1, weight=1)

window.configure(bg='#EBDEF0')

# Pencereyi açma ve çalıştırma
window.mainloop()
