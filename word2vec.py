import nltk,re,csv,random,math,pickle,time
from urllib.parse import unquote
from collections import Counter,deque
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import GeneSeg
learning_rate=0.1
vocabulary_size=3000
batch_size=128
embedding_size=128
num_skips=4
skip_window=3
valid_size=16
valid_window=100
top_k=8
num_sampled=64
num_steps=100001
plot_only=100
log_dir="word2vec.log"
plt_dir="file\\word2vec.png"
vec_dir="file\\word2vec.pickle"
start=time.time()

words=[]
with open("data\\xssed.csv","r",encoding="utf-8") as f:
    reader=csv.DictReader(f,fieldnames=["payload"])
    for row in reader:
        payload=row["payload"]
        word=GeneSeg(unquote(payload))
        words+=word
print("words size:",len(words))
#构建数据集
def build_dataset(words):
    count=[["UNK",-1]]
    counter=Counter(words)
    count.extend(counter.most_common(vocabulary_size-1))
    dictionary={}
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=[]
    for word in words:
        if word in dictionary.keys():
            data.append(dictionary[word])
        else:
            data.append(dictionary["UNK"])
            count[0][1]+=1
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    return count,data,dictionary,reverse_dictionary
count,data,dictionary,reverse_dictionary=build_dataset(words)
#生成训练Batch
data_index=0
def generate_batch(batch_size,num_skips,skip_window):
    '''
    :param batch_size: 生成的batch大小，必须为skip_window的整数倍
    :param num_skips: 对每个skip_window生成样本数量，不能大于skip_window*2
    :param skip_window: 目标单词取样本的窗口大小
    :return:
    '''
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=skip_window*2

    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1
    buffer=deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target=skip_window
        target_to_avoid=[skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target=random.randint(0,span-1)
            target_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0]=buffer[target]
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    return batch,labels
batch,labels=generate_batch(batch_size,num_skips,skip_window)
for i in range(100):
    print(batch[i],reverse_dictionary[batch[i]],"->",labels[i,0],reverse_dictionary[labels[i,0]])

valid_examples=np.random.choice(valid_window,valid_size,replace=False)
graph=tf.Graph()
with graph.as_default():
    with tf.name_scope("Inputs"):
        train_inputs=tf.placeholder(tf.int32,shape=[batch_size])
        train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
    with tf.name_scope("Embeddings"):
        embeddings=tf.Variable(
            tf.random_uniform(shape=[vocabulary_size,embedding_size],minval=-1.0,maxval=1.0),name="embeddings"
        )
        embed=tf.nn.embedding_lookup(embeddings,train_inputs)
    with tf.name_scope("nce_loss"):
        nce_weight=tf.Variable(
            tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size)),name="nce_weights"
        )
        nce_biases=tf.Variable(tf.zeros([vocabulary_size]),name="nce_biases")
        loss=tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                                           biases=nce_biases,
                                           labels=train_labels,
                                           inputs=embed,
                                           num_sampled=num_sampled,
                                           num_classes=vocabulary_size))

    with tf.name_scope("Train"):
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #标准化embeddings
        norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
        normalized_embeddings=embeddings/norm
    #计算验证数据与字典的相似性
    valid_embeddings=tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
    similarity=tf.matmul(
        valid_embeddings,normalized_embeddings,transpose_b=True
    )

    #tf.summary.histogram("nce_weight", nce_weight)
    #tf.summary.histogram("nce_biases", nce_biases)
    tf.summary.scalar("loss", loss)
    #tf.summary.histogram("normalized_embeddings",normalized_embeddings)
    merged=tf.summary.merge_all()
    init = tf.global_variables_initializer()
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized!")

    average_loss=0
    writer = tf.summary.FileWriter(log_dir, graph)
    for step in range(num_steps):
        batch_inputs,batch_labels=generate_batch(batch_size,num_skips,skip_window)
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
        loss_val,_,summary=session.run([loss,optimizer,merged],feed_dict=feed_dict)
        writer.add_summary(summary,global_step=step)
        average_loss+=loss_val
        if step%2000==0:
            if step>0:
                average_loss/=2000
                print("Average loss at step:",step,":",average_loss)
            average_loss=0
        if step%10000==0:
            if step>0:
                sim=similarity.eval()
                for i in range(valid_size):
                    valid_word=reverse_dictionary[valid_examples[i]]
                    nearest=(-sim[i,:]).argsort()[1:top_k+1]
                    log_str="Nearest to %s:"%valid_word
                    for k in range(top_k):
                        close_word=reverse_dictionary[nearest[k]]
                        log_str="%s %s"%(log_str,close_word)
                    print(log_str)
    final_embeddings=normalized_embeddings.eval()
    writer.close()
print(final_embeddings)

def plot_with_labels(low_dim_embs,labels,filename=plt_dir):
    plt.figure(figsize=(10,10))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),xytext=(5,2),
                     textcoords="offset points",
                     ha="right",
                     va="bottom")
        f_text="vocabulary_size=%d;batch_size=%d;embedding_size=%d;num_skips=%d;skip_window=%d;num_steps=%d"%(
            vocabulary_size,batch_size,embedding_size,num_skips,skip_window,num_steps
        )
        plt.figtext(0.03,0.03,f_text,color="green",fontsize=10)
    plt.show()
    plt.savefig(filename)
tsne=TSNE(perplexity=30,n_components=2,init="pca",n_iter=5000)

low_dim_embs=tsne.fit_transform(final_embeddings[:plot_only,:])
labels=[reverse_dictionary[i]for i in range(plot_only)]
plot_with_labels(low_dim_embs,labels)
def save(dictionary,reverse_dictionary,final_embeddings):
    word2vec={"dictionary":dictionary,"reverse_dictionary":reverse_dictionary,"embeddings":final_embeddings}
    with open(vec_dir,"wb") as f:
        pickle.dump(word2vec,f)
save(dictionary,reverse_dictionary,final_embeddings)
end=time.time()
print("Over job in ",end-start)