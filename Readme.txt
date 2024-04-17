用BERT訓練圍棋AI模型
===

## Motivation
自2016年 Alpha Go 擊敗李世石後，人工智慧在圍棋上的發展就突飛猛進。圍棋AI的主流作法是將圍棋盤面視為一張圖片，使用 ResNet 的相關模型去訓練。在圍棋中我發現兩件事:
1. 在AI眼中，盤面只是一張圖片，因此落子順序是沒差的；但是在人類眼中，不論如何避免，落子順序肯定會影響棋手。
2. 紀錄圍棋的`.sgf`檔案中，落子位置被轉化為一個個`[a-s][a-s]`的 token，並且以落子順序排列，形成一串句子。

因此我想知道，若將圍棋視作語言，會有什麼結果? 能否使用 BERT 來訓練圍棋 AI ?

---

## Dataset
### Source
* [Data1](http://sinago.com/qipu/new_gibo.asp): 職業比賽(about 33000 games)
* [Data2](https://github.com/featurecat/go-dataset): 野狐圍棋中九段玩家的棋局(about 140000 games)

### Definitions
* Train Data: 0~27000 games in Data1(sorted)
* Eval Data: 27000~30000 games in Data1(sorted)
* Test Data: 30000~33000 games in Data1(sorted)
* m: 一場棋局採用前幾步。
* Step-by-step data: 
    * 將一場棋局一步步拆開，使 (n,m) 的資料轉換成 (nxm,m) 的資料，範例如下:
    * `[...,[...],[5,4,3,2,1],[...],...]` ->`[...,[...],x1,x2,x3,x4,x5,[...],...]`
        * `x1 = [0,0,0,0,0]；y = 5`
        * `x2 = [5,0,0,0,0]；y = 4`
        * `x3 = [5,4,0,0,0]；y = 3`
        * `x4 = [5,4,3,0,0]；y = 2`
        * `x5 = [5,4,3,2,0]；y = 1`
* "Invalid move": 選擇的落子處已經有子存在

### Image Data
* ResNet 使用
* 使用一張(1,16,19,19)的圖片表示盤面(一張有16個 channels 的19x19圖片)
* 方法參考自 Alpha Go，16個 channel 包含:
    * white position(1)
    * black position(1)
    * empty position(1)
    * next turn(1)
    * last 6 moves(6)
    * stones have 1~6^+^ liberties(6)

### Language Data
* 361個位置形成361個token
* 使用BERT時加入 "SEP"，不用 "CLS"
* 使用一串(1,m)的句子表示盤面

### Difference between Language and Image
* 句子具有完整的棋局順序性；圖片則只有最後6步。
* 句子所需的記憶體容量小，只需(1,m)；圖片需要的記憶體容量則為(1,16,19,19)。
* 計算記憶體容量時，包含 token_id 以及 mask，兩者各半。

---

## Models
* 做一個 361 分類問題
* 設定三種模型大小
    * small: 1.3M parameters
    * mid: 4.4M parameters
    * large: 13M parameters

### ResNet
* 使用 Leela zero 的 ResNet 架構
* 使用 Conv2d 層時，加入足夠的 pad，維持圖片大小不變
* [Leela zero github](https://github.com/leela-zero/leela-zero)

### BERT
* 使用 Huggingface 的模型，自訂 vocab 大小等模型參數
* 使用 "mean of last hidden states" 來分類
---
(skip)
---

## Conclusion
### Performance
* 資料擁有相同記憶體容量下，隨著模型越大、資料越多，BERT 的 accuracy 趨近 ResNet
    * accuracy 已經幾乎追上
    * accuracy_5 還有差距
* 雖然 BERT 需要比較多筆資料，但是可以透過旋轉棋盤來解決
    * 在 m=240 下，需要的資料筆數約為6:1
    * 旋轉可以提供 4 倍
    * 再加上翻轉，總共可以提供 8 倍，但沒測過
* 旋轉棋盤的方法能提升 BERT 的 accuracy，但對 ResNet 似乎沒影響
    * 推測是因為 Image Data 已經包含相關位置的資料，但是 Language Data 沒有
    * 此方法進一步縮小了 BERT 跟 ResNet 的差距

### Advantage
* BERT 跟 ResNet 擅長的局面有差異，兩者可以互補
* BERT 的 accuracy 在資料及模型放大後，以及旋轉/翻轉棋盤的幫助下，有可能超越 ResNet

### Disadvantages
* BERT 訓練需要大量的時間
* 但隨著資料量增加，兩者需要的時間比例漸漸縮小

### Other things
* 將 Language Data 中的落子順序資訊移除，會大幅影響結果，顯示其重要性
* 將 token 代表的位置洗亂，不影響結果
* 即使 BERT 不知道棋盤上的空位，但在測試資料中，出現 invalid move 的機率跟 ResNet 一樣
* 透過 cosine 計算 token 的相似度觀察 token 間的關係
* 透過叫吃測驗，觀察 BERT 有沒有真正理解各 token 的意義及位置關係
