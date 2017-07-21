from utils import GeneSeg
import csv,pickle,random
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

vec_dir="file\\word2vec.pickle"
process_datas_dir="file\\process_datas.pickle"

def pre_process():
    with open(vec_dir,"rb") as f :
        word2vec=pickle.load(f)
        dictionary=word2vec["dictionary"]
        reverse_dictionary=word2vec["reverse_dictionary"]
        embeddings=word2vec["embeddings"]
    xssed_data=[]
    normal_data=[]
    with open("data\\xssed.csv","r",encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=["payload"])
        for row in reader:
            payload=row["payload"]
            word=GeneSeg(payload)
            xssed_data.append(word)
    with open("data\\normal_examples.csv","r",encoding="utf-8") as f:
        reader=csv.reader(f)
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
    labels=to_categorical(labels)
    def to_index(data):
        d_index=[]
        for word in data:
            if word in dictionary.keys():
                d_index.append(dictionary[word])
            else:
                d_index.append(dictionary["UNK"])
        return d_index
    datas_index=[to_index(data) for data in datas]
    datas_index=pad_sequences(datas_index,value=-1)
    rand=random.sample(range(len(datas_index)),len(datas_index))
    datas=[datas_index[index] for index in rand]
    labels=[labels[index] for index in rand]
    train_datas,test_datas,train_labels,test_labels=train_test_split(datas,labels,test_size=0.3)
    process_datas=word2vec
    process_datas["train_datas"]=train_datas
    process_datas["test_datas"]=test_datas
    process_datas["train_labels"]=train_labels
    process_datas["test_labels"]=test_labels
    with open(process_datas_dir,"wb") as f:
        pickle.dump(process_datas,f)
    print("Preprocessing data over!")
    print("Saved datas and labels to ",process_datas_dir)
def build_dataset(batch_size):
    with open(process_datas_dir, "rb") as f:
        process_datas = pickle.load(f)
    embeddings = process_datas["embeddings"]
    dictionary = process_datas["dictionary"]
    reverse_dictionary = process_datas["reverse_dictionary"]
    train_datas = process_datas["train_datas"]
    train_labels = process_datas["train_labels"]
    test_datas = process_datas["test_datas"]
    test_labels = process_datas["test_labels"]
    dims_num = embeddings["UNK"].shape[0]
    input_num = len(train_datas[0])
    def batch_generator(datas, labels,batch_size,train=True):
        batch_data = []
        batch_label = []
        while True:
            for index in range(len(datas)):
                data_embed = []
                for d in datas[index]:
                    if d != -1:
                        data_embed.append(embeddings[reverse_dictionary[d]])
                    else:
                        data_embed.append([0.0] * len(embeddings["UNK"]))
                batch_data.append(data_embed)
                batch_label.append(labels[index])
                if len(batch_label) == batch_size:
                    yield (np.array(batch_data), np.array(batch_label))
                    batch_data = []
                    batch_label = []
            if not train and len(batch_label)>0:
                yield (np.array(batch_data), np.array(batch_label))
                break
    train_generator = batch_generator(train_datas, train_labels,batch_size)
    test_generator = batch_generator(test_datas, test_labels,batch_size,train=False)
    train_size=len(train_labels)
    test_size=len(test_labels)
    return train_generator,test_generator,train_size,test_size,input_num,dims_num
if __name__=="__main__":
    pre_process()





