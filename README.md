# Transformer in Computer Go

---
## Introduction

### Computer Go Introduction
* The first AI to defeat top human players is AlphaGo.
* AlphaGo = "**Policy head**" + "Value head" + "RL"
    * Policy head: Choose several positions for value head.
    * Value head: Determine whether a situation is good or not.
    * Policy head + Value head: Search many positions(like a tree) to choose the best one.
    * Value head + RL: Model can learn itself and know things not in training data.
* The policy head is implemented by Resnet tranditionally.
* I want to know whether we can **use Transformer instead if Resnet in policy head**.

### Goals
1. Compare ResNet, Vitual Transformer, and Swim Transformer. They are image models.
2. See Go's game as a sequence, then use BERT to train a "language model".
3. Main evaluate way is "compare to Resnet"
4. Try to analyze by some Go's knowledge.

---

## Data

### Introduction
* Data1:Professional game records(about 35000 games)
* Data2:Foxwq 9d's game records(about 140000 games)
* 12 first steps
    * Choose the 12 most common first move for models, otherwise they will always generate same games.
    * first_steps:["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", "pc", "pp", "pq", "qp"] 
    * Data1: 34541/34931；99% games' first steps are in my list.
    * Data2: 142353/143458；99% games' first steps are in my list.

### Process
1. Get n games.
2. Check they are valid.
3. Take first m moves.
4. Tokenize data.
5. Split into trainData(90%), evalData(10%)
6. Step-by-step data.   
    A (n,m) shape data will become a (nxm,m) shape data.
    * ex:`[...,[5,4,3,2,1],...]` will become`[...,x1,x2,x3,x4,x5,...]`
    * `x1 = [0,0,0,0,0]；y = 5`
    * `x2 = [5,0,0,0,0]；y = 4`
    * `x3 = [5,4,0,0,0]；y = 3`
    * `x4 = [5,4,3,0,0]；y = 2`
    * `x5 = [5,4,3,2,0]；y = 1`
7. Picture Data: change each data to a picture.  
   A (nxm,m) shape data will become a (nxm,16,19,19) shape data.
8. Sequence Data: add "CLS" and "SEP".  
   A (nxm,m) shape data will become a (nxm,m+2) shape data.

### Picture Data
* Convert a 19x19 board to a 16x19x19 tensor, which means a picture having 16 channels
* 16 channels consist of:
    * black position
    * white position
    * empty position
    * next turn
    * last 6 moves
    * 1~6^+^ liberties of the stones

### Sequence Data
* 361 positions, "CLS" token, "SEP" token form a 363 words vocab.
* Use a sequence to represent games.

### Rotation and Reverse
* Because board is a square, so a board can have 8 same situation.
    * Rotation has 4 same situation(360/90 = 4)
    * Reverse has 2 same situation
    * 4x2 = 8
* We can let all games have same situation when games start by:
    * Let first move in top_left's 1/4 square region
    * Let second move in top_right's 1/2 triangle region

---

## Evaluation
* Accuracy
    * Obtain by eval data when training
* Error rate
    * Model cannot output a position already having a stone.
    * Let models play with itself or other model.
    * Scores:
        * score: x games would not have any error before 80/160 moves.
        * moves:  model averagely would not have any error before x moves.
        * error: model have error x times in games with each other.

* Average Distance
    * In Go, we cannot focus on a small region too more or too less.
    * Caculate the rate that each move is near its last move.

* Atari rate
    * Caculate the rate that each move "atari" opponent.

* Liberty
    * Caculate each stone's liberty when it is played.
    


---

## Models

### Introduction
* I adjust models' paramaters to make them have similar sizes.
* Each model has three sizes, model0, model1, model2, and model0 > model1 > model2.
    * XXX0: about 13500000 paramaters
    * XXX1: about 4400000 paramaters
    * XXX2: about 1300000 paramaters 
* Same index means similar size, ex: ResNet1 ~= ViT1 ~= ST1 ~= BERT1

### ResNet
Use Leela zero's model architecture.

### Visual Transformer
Use 2 Conv2d layers, huggingface's Visual Transformer model and some linear model.

### Swim Transformer
Use 1 Conv2d layer, huggingface's Swim Transformer model and some linear model.

### BERT
Use huggingface's BERT model and some linear model.



---

## Results and Discussions

### Rotation and Reverse
> moves: 80
> data: data1

![image](https://hackmd.io/_uploads/HJeK62bha.png =70%x)
* In all models, rotation and reverse seem not important.
* So I don't use rotation and reverse in following training.

### BERT
#### Same Memory Size
* The main difference between BERT and other models is that BERT's data is "sequence".
* An image's shape is (1,16,19,19), but a sequence is only (1,80).
* So under same memory size, BERT can have lagerer data size.
> moves: 80
> data: data1

![image](https://hackmd.io/_uploads/HJJJoJ0iT.png =60%x)
> moves: 80
> data: data2

![image](https://hackmd.io/_uploads/ByyejyCoT.png =60%x)
* BERT model can work and have good performance if we **have huge amount of data**.
* But BERT **need lots of time** to train because large data size, and it also need much more epochs.

#### Score
> moves: 80
> data: data1  

![image](https://hackmd.io/_uploads/ryIoUMf26.png =80%x)
* Although BERT has high accuracy, but it has high error and low score and moves.
* It means that BERT often chooses invalid position.

#### BERT learns relative poisition of words
* Atari test:
    * After move_17(black), move_18(white) is almost the only possible option of next move.
    * Because in Go, it would be a very bad situation if move_14 is eaten.
    * This situation is abnormal, so training data does not have similar data.
    * But BERT still know move_18 is important.
    * Although I exchange move_13, move_15, move_17 in 6 kinds(3! = 6), they all have same answer.
        * (9, 10) is move_18.
        * Probability: `high->low：left->right.`
        * `[(8, 9), (10, 8), (10, 10), (8, 8), (9, 10)]`
        * `[(8, 9), (10, 8), (10, 10), (8, 8), (9, 10)]`
        * `[(10, 8), (9, 8), (10, 10), (8, 8), (9, 10)]`
        * `[(10, 8), (9, 8), (10, 10), (8, 8), (9, 10)]`
        * `[(9, 8), (8, 9), (8, 8), (10, 10), (9, 10)]`
        * `[(7, 9), (8, 9), (10, 10), (8, 8), (9, 10)]`
    * **How BERT know move_14 only has only one liberty and choose move_18?**
    * move_1~17: ['cd','dq','pq','qd','oc','qo','co','ec','de','pe','np','fp','kj','jj','ji','ql','ij']
![image](https://hackmd.io/_uploads/BJsTF1Ks6.png =70%x)

#### Pretrain?

---

### Transformer vs ResNet
#### Score
Data sizes:[500, 1600, 5000, 15000, 30000]
Accuracy:

Accuracy_5:

Moves:




---


















Todo:
Simple GUI
實力判斷
