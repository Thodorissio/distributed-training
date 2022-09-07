import numpy as np 
import pandas as pd 
import tensorflow as tf
import sklearn
from tqdm import tqdm

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

def convert_data_to_examples(train, test, val, review, sentiment): 
    train_InputExamples = train.apply(lambda x: InputExample(
                                                            guid=None, 
                                                            text_a = x[review], 
                                                            label = x[sentiment]), 
                                                            axis = 1,
                                                        )

    test_InputExamples = test.apply(lambda x: InputExample(
                                                            guid=None, 
                                                            text_a = x[review], 
                                                            label = x[sentiment]), 
                                                            axis = 1,
                                                        )
    
    validation_InputExamples = val.apply(lambda x: InputExample(
                                                            guid=None, 
                                                            text_a = x[review], 
                                                            label = x[sentiment]), 
                                                            axis = 1,
                                                        )
  
    return train_InputExamples, test_InputExamples, validation_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] 

    for e in tqdm(examples):
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,    
            max_length=max_length,    
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, 
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],input_dict["token_type_ids"], input_dict['attention_mask'])
        features.append(InputFeatures( input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label) )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
def cat2num(value):
    if value=='positive': 
        return 1
    else: 
        return 0
    

def imdb_dataset():
  df=pd.read_csv("/content/drive/MyDrive/IMDB Dataset.csv")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  df['sentiment']  =  df['sentiment'].apply(cat2num)
  train = df[:450] #kanonika train = df[:45000]
                    # test = df[45000:49000]
                    #val = df[49000:]
  test = df[450:490]
  val = df[490:550]
  train_InputExamples, test_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, val, 'review',  'sentiment')
  train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
  train_data = train_data.shuffle(100).batch(32).repeat(2)
  test_data = convert_examples_to_tf_dataset(list(test_InputExamples), tokenizer)
  test_data = test_data.batch(32)
  validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
  validation_data = validation_data.batch(32)
  
  return train_data, test_data , validation_data



def build_and_compile_cnn_model():
  model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
  for layer in model.layers[:-1]:  #analoga me poses parametrous ekapideusimes theloume -poso bari mporoume to modelo
    layer.trainable=False
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

  return model



