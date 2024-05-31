利用 BERT 之圍棋 AI 下子預測
BERT for Go Prediction
===



## Dataset
### Source
* Data1:職業比賽(about 33000 games)
* Data2:野狐圍棋中九段玩家的棋局(about 140000 games)

### Definitions
* Train Data: 0~27000 games in Data1
* Eval Data: 27000~30000 games in Data1
* Test Data: Data2
* m: 一場棋局採用前幾步。
* Step-by-step data: 
    * 將一場棋局一步步拆開，使 (n,m) 的資料轉換成 (nxm,m) 的資料，範例如下:
    * `[...,[...],[5,4,3,2,1],[...],...]` ->`[...,[...],x1,x2,x3,x4,x5,[...],...]`
        * `x1 = [0,0,0,0,0]；y = 5`
        * `x2 = [5,0,0,0,0]；y = 4`
        * `x3 = [5,4,0,0,0]；y = 3`
        * `x4 = [5,4,3,0,0]；y = 2`
        * `x5 = [5,4,3,2,0]；y = 1`
    * 也就是說，使用 n 盤，會產生 nxm 筆資料
* Invalid move: 選擇的落子處已經有子存在
* Accuracy_n: 機率最高的前 n 選有選中的比率

### ResNet Data
* 輸入(1,channel,19,19)的圖片表示一個盤面
* 資料處理方式模仿 Alpha Go，但簡化內容使其記憶體容量跟 BERT Data 一樣，其channels 包含: 
    * white position(1)
    * black position(1)
    * next turn(1)
    * stones have 1~6^+^ liberties(1)(棋子的"氣")

### BERT Data
* 361個位置形成361個 token
* 加入 "SEP" 在結尾
* 輸入為三個 (1,m) 的 tensor，包含:
    * input_ids(落子位置順序)
    * masks(就是 mask)
    * token_type_ids(棋子的"氣")
* 將 token_types_ids 當成 embedding 資訊的管道
* 若要新增其他資訊，就使用相同方法，將其放入 embedding 層

## Models
* 做一個 361 分類問題
* 模型大小: 4.4M parameters
* 皆沒有預訓練

### ResNet
* 使用 ResNet 架構

### BERT
* 使用 Huggingface 的模型，自訂模型參數
* 使用 mean of last hidden states 來分類
* 特殊參數設定:
```python!
config = BertConfig()
config.type_vocab_size = 7
config.vocab_size = 363
config.num_attention_heads = 1
config.position_embedding_type = "relative_key"
```

---
