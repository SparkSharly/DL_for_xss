import time
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from utils import GeneSeg
import csv,random,pickle


batch_size=50
maxlen=200
vec_dir="file\\word2vec.pickle"
epochs_num=1
log_dir="log\\MLP.log"
model_dir="file\\SVM_model"
def pre_process():
    with open(vec_dir,"rb") as f :
        word2vec=pickle.load(f)
        dictionary=word2vec["dictionary"]
        embeddings=word2vec["embeddings"]
        reverse_dictionary = word2vec["reverse_dictionary"]
    xssed_data=[]
    normal_data=[]
    with open("data\\xssed.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            xssed_data.append(word)
    with open("data\\normal_payload.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            normal_data.append(word)
    xssed_num=len(xssed_data)
    normal_num=len(normal_data)
    xssed_labels=[1]*xssed_num
    normal_labels=[0]*normal_num
    datas=xssed_data+normal_data
    labels=xssed_labels+normal_labels
    def to_index(data):
        d_index=[]
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index
    datas_index=[to_index(data) for data in datas]
    datas_index=pad_sequences(datas_index,value=-1,maxlen=maxlen)
    rand=random.sample(range(len(datas_index)),len(datas_index))
    datas=[datas_index[index] for index in rand]
    labels=[labels[index] for index in rand]
    datas_embed=[]
    dims=len(embeddings["UNK"])
    n=0
    for data in datas:
        data_embed = []
        for d in data:
            if d != -1:
                data_embed.extend(embeddings[reverse_dictionary[d]])
            else:
                data_embed.extend([0.0] * dims)
        datas_embed.append(data_embed)
        n+=1
        if n%1000 ==0:
            print(n)
    train_datas,test_datas,train_labels,test_labels=train_test_split(datas_embed,labels,test_size=0.3)
    return train_datas,test_datas,train_labels,test_labels
if __name__=="__main__":
    train_datas, test_datas, train_labels, test_labels=pre_process()
    print("Start Train Job! ")
    start = time.time()
    model=LinearSVC()
  #  model = SVC(C=1.0, kernel="linear")
    model.fit(train_datas,train_labels)
   # model.save(model_dir)
    end = time.time()
    print("Over train job in %f s" % (end - start))
    print("Start Test Job!")
    start=time.time()
    pre=model.predict(test_datas)
    end=time.time()
    print("Over test job in %s s"%(end-start))
    precision = precision_score(test_labels, pre)
    recall = recall_score(test_labels, pre)
    print("Precision score is :", precision)
    print("Recall score is :", recall)
    with open(model_dir,"wb") as f:
        pickle.dump(model,f,protocol=2)
    print("wirte to ",model_dir)