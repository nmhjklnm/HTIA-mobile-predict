# HTIA: An Intelligent Mobile Prediction System

HTIA is an intelligent mobile prediction system that utilizes deep learning techniques to accurately predict individual mobility patterns. The prediction of individual mobility has been shown to hold significant commercial and social value in traffic planning and location advertising.![fig2](https://nmhjklnm.oss-cn-beijing.aliyuncs.com/article-img/img/fig2.svg)

## Features

The HTIA system currently supports the following features:

- **Advanced Deep Learning Model**: HTIA incorporates a novel sequence-to-sequence (seq2seq) model with mini-batch hierarchical temporal incidence attention (HTIA) to capture long-term and short-term dependencies in individual mobility patterns.
- **Improved Prediction Accuracy**: Our approach surpasses state-of-the-art competing schemes, reducing mean relative error by more than 70.8%, 60.8%, and 69.9% respectively, as demonstrated in extensive experiments conducted on three public datasets exhibiting different degrees of uncertainty.
- **Efficiency and Interpretability**: We enhance the efficiency of the model by employing sequence padding and incorporating it into HTIA, while maintaining its interpretability.

## supplymentary material  manual

---

### Directory Tree

```python
│  readme.md #please read it firstly
│  readme.pdf #readme .pdf version
│  requirements.txt # python>=3.9.0 summarize the virtual enviroments of this package
│  summarize.ipynb # store any  visualization result of codes and picture 
│  summarize.pdf  # the pdf of summarize.ipynb
├─common experiment #Overall performerance
│  ├─HTIA           #the main model of this project
│  │  │  config.py  #config 
│  │  │  eval.py    #stored eval function
│  │  │  main.py    #HTIA main procedures
│  │  │  requirements.txt # same with above
│  │  │  train.py   #train function
│  │  │  __init__.py
│  │  ├─data        #stored data processing function
│  │  │  │  dataset.py
│  │  │  │  num_sequence.py #Geographic information coding
│  │  │  │  __init__.py
│  │  ├─decoder
│  │  │  │  attention.py   #all attenion stored in there
│  │  │  │  decoder_model.py
│  │  │  │  __init__.py
│  │  │  │  
│  │  ├─dic  #completely Geographic information encoding dictionary 
│  │  │      981762.pkl
│  │  │      981808.pkl
│  │  │      981814.pkl
│  │  ├─encoder 
│  │  │  │  embedding_concat.py
│  │  │  │  encoder_model.py
│  │  │  │  __init__.py
│  │  ├─models #Store the trained model only display trained manuseed==2
│  │  │  ├─981762   
│  │  │  ├─981808
│  │  │  └─981814    
│  │  ├─runs #experiment log
│  │  ├─seq2seq
│  │  │  │  seq2seq_model.py 
│  │  └─raw_data #raw_data
│  │      │  981762.txt
│  │      │  981808.txt
│  │      │  981814.txt    
│  ├─HTIA-UP # other model ，This folder only stores the code, not the trained model
│  ├─HTAED-GRU#But you can cd to a folder and use tensorboard to view the results
├─hyperparameter experiment 
│  ├─HTIA-embedding ##different embedding，This folder only stores the code
│  ├─HTIA-head#But you can cd to a folder and use tensorboard to view the results
```

## Getting Started

### optional 1：directly test run

Open a terminal and run:

```python
#python>=3.9.0 summarize the virtual enviroments of this package
pip install -r requirements.txt
cd common experiment\HTIA
python main.py --mode eval
```



### optional 2:full training

Open a terminal and run:

```python
pip install -r requirements.txt
cd common experiment\HTIA
python main.py --mode train
# or python main.py --mode train # If you want to train yourself 
#(run for train 3datasets and each 5 different maunal seed)
# optional untrain or eval  ,you can directly check results by tensorborad 
tensorboard --logdir=runs
```

### View experiment results directly

```python
cd <cd to the corresponding model folder>
tensorborad --logdir=runs
```

## License 

The HTIA system is licensed under the MIT License. See the LICENSE file for more information.

