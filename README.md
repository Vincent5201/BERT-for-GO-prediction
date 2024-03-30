Transformer in Computer Go
---

## Abstract
### Introduction
* The first AI to defeat top human players is AlphaGo.
* AlphaGo = "Policy head" + "Value head" + "RL"
    * Policy head: Choose several positions for value head.
    * Value head: Score a situation.
    * Policy head + Value head: Search for the best positions.
    * Value head + RL: Model can learn itself.
* The policy head is implemented by Resnet tranditionally.
* I want to use Transformer instead of ResNet in Policy head and see what would happen.

### Goal
* Use Visual Transformer and Swim Transformer to compare with ResNet.
* Compare BERT(language model) and other image models.
* Analyze how BERT treat Go.
* Make a small Go AI.
___

## Models
Let all models have about 4400000 paramaters.
### ResNet
* Copy Leela zero's model architecture.
* Image size would not be reduced in Conv2d layers beacuse of enough padding.
* [Leela zero github](https://github.com/leela-zero/leela-zero)

### Visual Transformer(ViT)
* Use huggingface's Visual Transformer model.
* Add 2 Conv2d layers can improve performance.
* Use linear layer do pool can improve performance.

### Swim Transformer(ST)
* Use huggingface's Swim Transformer model.
* Use 1 Conv2d layer can improve performance.
* Output dim < 361, so use linear layer to expand can improve performance.

### BERT
* Use huggingface's BERT model.
* Use mean of output can improve performance.
* Can use pretrained BERT.

## Data
### Source
* [Data1](http://sinago.com/qipu/new_gibo.asp): Professional game records(about 35000 games)
* [Data2](https://github.com/featurecat/go-dataset): Foxwq 9d's game records(about 140000 games)

### Definitions
* Train Data: 0~27000 games in Data1(sorted)
* Eval Data: 27000~30000 games in Data1(sorted)
* Test Data: 30000~ games in Data1(sorted)
* game: Each game contains hundreds of moves in order.
* move: A move is a position on the board, represented in `[a-s][a-s]`.
* invalid move: A move cannot choose a position where another stone is there.
* 12 first moves:
    * The most common first moves.
    * they are: `["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", "pc", "pp", "pq", "qp"]`
    * Data1: 34541/34931；99% games' first steps are in my list.
    * Data2: 142353/143458；99% games' first steps are in my list.
* A correct beginning: top four moves are
    * No invalid move.
    * All in "first steps".
    * Belongs to four different corners respectively.
* Step-by-step data: 
    * Transfer a (n,m) shape data into a (nxm,m) shape data.
    * Let a game become m games.
    * ex:`[...,[...],[5,4,3,2,1],[...],...]` will become`[...,[...],x1,x2,x3,x4,x5,[...],...]`
    * `x1 = [0,0,0,0,0]；y = 5`
    * `x2 = [5,0,0,0,0]；y = 4`
    * `x3 = [5,4,0,0,0]；y = 3`
    * `x4 = [5,4,3,0,0]；y = 2`
    * `x5 = [5,4,3,2,0]；y = 1`
* m: I choose 240 because generally speaking, in Go, after 240 moves, it is obvious who wins.

### Image Data
* Used by Visual Transformer, Swim Transformer and ResNet.
* Transfer each (1,m) shape data into a (1,16,19,19) shape data.
* Represents informations on the board, referenced from leela zero.
* 16 channels consist of:
    * black position(1)
    * white position(1)
    * empty position(1)
    * next turn(1)
    * last 6 moves(6)
    * 1~6^+^ liberties of the stones(6)

### Language Data
* Tokens including 361 positions, "CLS", "SEP", and "MAK"(for pretrain).
* Transfer each (1,m) shape data into a (1,m+2) shape data.
* Use a sequence to represent a game.

### Pretrain Data(for BERT)
* Masked Language Modeling(MLM): 15% tokens are masked.
* Next Sentence Prediction(NSP): 
    * Let games with same number of padding be a group
    * Choose 50% of games in same group to random swap half of themselvs.
    * So 50% of total games' NSP labels will be "false". 
* Transfer each (1,m) shape data into a (1,m+3) shape data.

### Data Processing
1. Get n games.
2. Check they are valid.
3. Take first m moves.
4. Tokenize data.
5. Split into trainData(90%), evalData(10%)
6. Step-by-step data. 
7. Transfer to image Data/language Data/pretrain Data

___

## Evaluation
* Basic standard is ResNet. 
* Accuracy
    * Accuracy: first choice is correct
    * Accuracy_5/10: top 5/10 choices contain correct choice
* Valid rate
    * valid_score: Let model plays with itself, x games would not have any error before m moves.
    * valid_moves: model averagely would not have any invalid move before x moves.
* rank: Play with foxwq AI to realize its approximate ability.
___
