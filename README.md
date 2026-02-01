# F025pro GNN+知识图谱图书推荐系统vue+flask+neo4j架构 （两种基于知识图谱的推荐算法（GNN和基于路径推荐）+基于用户协同过滤（总共三种推荐算法））

> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从git来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
关注B站，有好处！
> 

**编号: F025 pro**

架构: vue+flask+neo4j+mysql+pytorch
亮点：两种基于知识图谱的推荐算法（GNN和基于路径推荐）+基于用户协同过滤（总共三种推荐算法）
支持爬取图书数据，数据超过万条，知识图谱节点几万个

## 视频演示

[video(video-oPcJS6bR-1745713829508)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=114402040746557)(image-https://i-blog.csdnimg.cn/img_convert/46c26631a36e691bcc678ffe18867372.jpeg)(title-基于GNN图书知识图谱推荐系统（vue+flask+neo4j+mysql）|图神经网络推荐+可视化)]
## 使用算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8338ad1d5c84418e929bb256bd00e134.png)
## 架构说明
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b235c19ac874044a8e89a3c172a799b.png)
## 功能模块
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8d17e950e8c149dba68c3af8ca5a876a.png)

系统架构主要分为以下几个部分：**用户前端**、**后端服务**、**数据库**、**数据爬取与处理**。各部分通过协调工作，实现数据的采集、存储、处理以及展示。具体如下：
### 1. 用户前端
**用户**通过浏览器访问系统，前端采用了基于 Vue.js 的技术栈来构建。
- **浏览器**：作为用户与系统交互的媒介，用户通过浏览器进行各种操作，如浏览图书、获取推荐等。
- **Vue 前端**：使用 Vue.js 框架搭建前端界面，包含 HTML、CSS、JavaScript，以及 Vuex（用于状态管理），vue-router（用于路由管理），和 Echarts（用于数据可视化）等组件。前端向后端发送请求并接收响应，展示处理后的数据。
### 2. 后端服务
后端服务采用 Flask 框架，负责处理前端请求，执行业务逻辑，并与数据库进行交互。
- **Flask 后端**：使用 Python 编写，借助 Flask 框架处理 HTTP 请求。通过 SQLAlchemy 与 MySQL 进行交互，通过 py2neo 与 Neo4j 进行交互。后端主要负责业务逻辑处理、 数据查询、数据分析以及推荐算法的实现。
### 3. 数据库
系统使用了两种数据库：关系型数据库 MySQL 和图数据库 Neo4j。
- **MySQL**：存储从网络爬取的基本数据。数据爬取程序从外部数据源获取数据，并将其存储在 MySQL 中。MySQL 主要用于存储和管理结构化数据。
- **Neo4j**：存储图谱数据，特别是用户、图书及其关系（如阅读、写、出版等）。通过利用 py2neo 库将 MySQL 中的数据结构化为图节点和关系，再通过图谱生成程序（可能是一个 Python 脚本）将其导入到 Neo4j 中。
### 4. 数据爬取与处理
数据通过爬虫从外部数据源获取，并存储在 MySQL 数据库中，然后将数据转换为图结构并存储在 Neo4j 中。
- **爬虫**：实现数据采集，从网络数据源抓取相关信息。爬取的数据首先存储在 MySQL 数据库中。
- **图谱生成程序**：利用 py2neo 将爬取到的结构化数据（如用户、图书、作者、出版社，以及它们之间的关系）从 MySQL 导入到 Neo4j 中。通过构建图谱数据，使得后端能够进行复杂的图查询和推荐计算。
### **工作流程**
1. **数据爬取**：爬虫程序从外部数据源抓取数据并存储到 MySQL 数据库中。
2. **数据处理与导入**：图谱生成程序将 MySQL 中的数据转换为图结构并导入到 Neo4j 中，利用 py2neo 与 Neo4j 交互。
3. **前后端交互**：
    - 用户通过浏览器访问系统，前端用 Vue.js 构建，提供友好的用户界面和交互。
    - 前端向 Flask 后端发送请求，获取图书信息或推荐图书。
4. **推荐算法**：后端在接收请求后，利用 Neo4j 图数据库中的数据和关系进行处理（如推荐计算），并使用 py2neo 库与 Neo4j 交互获取数据结果。
5. **数据返回与展示**：后端将计算结果返回给前端进行展示，通过 Vue.js 的图表库（如 Echarts）进行数据可视化，让用户得到直观的推荐结果和分析信息。
### **小结**
这套系统通过整合爬虫、关系型数据库、图数据库，以及前后端的协调配合，实现了数据的高效采集、存储、处理、推荐和展示。从用户体验的角度，系统能够提供高度个性化的推荐，并通过图形化的方式呈现数据分析结果。
## 功能介绍
### 0 图谱构建
利用python读取数据并且构建图谱到neo4j中
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9feb39fef5f74a7297e785ba46107920.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13fa1416e4c84ec0a02d5a93a1986107.png)
### 1 系统主页，统计页面
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/335e32cf2d4f4d4b95233eb05f889419.png)
### 2 知识图谱
支持可视化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b8a54d4aa454c5999df64c40f947c51.png)
支持模糊搜索，比如搜索法国作家 加缪
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7eaa5ed10895430690d90c052542d615.png)
### 3 推荐算法
没有登录无法推荐
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/489a92d9ed5f4864a5c589de73c7ecf4.png)
**基于GNN 图神经网络训练和推荐**
合并特征、GNN训练、基于路径的推荐
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/214556ec7de14d168c87df829869cef7.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/808e2af1fafa4e85bac5cac895cbefce.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/908bcd30b7cb4d24882f6644bd2eb755.png)
基于知识图谱路径的推荐算法
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7479df1a0c544c8cb514c898246630ac.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/72abf8f66d8b4389b38689c31dbdc576.png)

基于用户协同过滤推荐算法推荐
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5f4dc074e6c94ba5888f8ece9625c2ca.png)

点击可以进入图书详情页面(可以查看 名称、作者、系列、图片、**装帧、用户给图书的评分）**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9f8913f217ae4fc0ae1e8009cec740f5.png)
支持使用评分控件进行评分
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7a4e52060f8844969342369b635bfd8d.png)
### 4 可视化分析
分为4个页面
图书出版地图分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9c33da0cf798458eba0a2085ded994d1.png)
图书分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e0a43d21937c495aa585b5df614dc1c6.png)
图书评分分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6c023e55b31e4dc2a8c183d1e7c51ea2.png)
图书词云分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9f9dcf50d8c8417485cef644ef4a00c2.png)
### 5 登录与注册
支持登录与注册
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71c9fc71d24847e9a6f74f87418a4c57.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fd4c907b310845e382c3812ac56474a3.png)
## 算法
### 算法介绍
该算法首先通过构建图书知识图谱，整合图书、作者、类别和用户的关系数据。然后利用GAT（图注意力网络）进行图嵌入学习，生成用户和图书的低维表示向量。通过计算用户与图书之间的余弦相似度，生成个性化推荐。模型采用PyTorch Lightning框架，支持分布式训练和批量处理，适用于大规模图数据。
### 算法流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/87c178d9da62479aa3b83e9df4e2ab4f.png)

### 核心代码
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.lightning import LightningModule
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader

# 定义图书推荐模型
class BookRecommender(LightningModule):
    def __init__(self, num_nodes, num_relations, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.book_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.author_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.category_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.gnn = GATConv(embedding_dim, embedding_dim)
        
    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.user_embedding(x)
        return self.gnn(x, edge_index)

# 定义数据加载器
class BookDataset(Dataset):
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return Data(x=self.graph_data[idx]['x'], 
                    edge_index=self.graph_data[idx]['edge_index'], 
                    edge_type=self.graph_data[idx]['edge_type'])

# 训练模型
def train_model(model, dataloader, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

# 生成推荐
def generate_recommendations(model, user_id, k=5):
    user_node = torch.tensor([user_id])
    with torch.no_grad():
        user_emb = model.user_embedding(user_node)
    similarities = []
    for book_id in all_books:
        book_emb = model.book_embedding(torch.tensor([book_id]))
        similarity = torch.cosine_similarity(user_emb, book_emb)
        similarities.append((book_id, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in similarities[:k]]


```
