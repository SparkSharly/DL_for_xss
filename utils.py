import nltk
import re
from urllib.parse import unquote
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
def GeneSeg(payload):
    #数字泛化为"0"
    payload=payload.lower()
    payload=unquote(unquote(payload))
    payload,num=re.subn(r'\d+',"0",payload)
    #替换url为”http://u
    payload,num=re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    return nltk.regexp_tokenize(payload, r)
def init_session():
    #gpu_options=tf.GPUOptions(allow_growth=True)
    ktf.set_session(tf.Session())