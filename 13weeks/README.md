# 13주차 (LightGCN 분석, 순차적 추천 LSTM)

### Light GCN 코드 실행

```bash
pip install LibRecommender
#-- https://github.com/massquantity/LibRecommender#references 라이브러리
```

 ****

**scatter와 sparse 라이브러리를 다운로드 하기 위해서는 컴퓨터의 cuda 버전을 확인하고 그 버전에 맞는 라이브러리를 다운로드해야 합니다.**

```python
print(torch.__version__
print(torch.version.cuda)
#-- 위 명령어로 컴퓨터의 Cuda 버전을 확인 colab은 cu118을 사용하고 있으므로 cu118.html사용
2.0.1+cu118
11.8
```

버전 확인 후 라이브러리 다운로드

```bash
pip install torch-geometric
pip install -q git+https://github.com/snap-stanford/deepsnap.git
"""
DeepSNAP는 그래프 신경망(Graph Neural Networks)을 구축하고 훈련하기 위한 파이썬 패키지
DeepSNAP은 PyTorch 기반의 그래프 신경망 라이브러리 PyG(PyTorch Geometric)에서 영감을 받아 개발되었다. DeepSNAP은 PyG와 유사한 인터페이스를 제공하여 그래프 데이터를 로드하고 전처리할 수 있다.
"""
pip install -U -q PyDrive
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

1. **필요 라이브러리 import(여기서부터 파이썬 코드)** 

```python
import numpy as np
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import LightGCN  # pure data, algorithm NGCF
from libreco.evaluation import evaluate
#-- Libreco 라이브러리 LightGCN을 위한 import

import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim, Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.data import download_url, extract_zip
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
#-- import required modules Blog LightGCN을 위한 import 
```

1. **데이터 준비**
- 데이터를 다운받고 압축해제 후 ratings.csv와 movies.csv의 path를 저장

```python
# download the dataset
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
extract_zip(download_url(url, '.'), '.')

rating_path = '/content/ml-latest-small/ratings.csv'
item_path = "/content/ml-latest-small/movies.csv"
```


<details>
<summary> 라이브러리 설치를 제외한 통코드이며 데이터를 다운로드 후 패스 지정까지 했다고 가정 </summary>
<div markdown="1">   

```python
#-- 데이터 전처리

def libco_LightGCN_data(path, data_frame, ratio = [0.8, 0.1, 0.1], excluded_user=None):
  #-- data
  data = pd.read_csv(path)
  data.rename(columns = {data_frame[0] : "user", data_frame[1]:"item", data_frame[2]:"label",},
              inplace = True)
  
  # print(data.head(10))
  #-- 테스트를 위한 특정 유저의 가장 앞에 있는 데이터 n개 삭제
  if excluded_user is not None:
    user_data = data[data["user"] == excluded_user]
    user_data_5stars = user_data[user_data["label"] == 5.0]  # Filter the data with a rating of 5.0
    if not user_data_5stars.empty:
      excluded_number = 15
      user_data_5stars = user_data_5stars.head(excluded_number)
      data = data.drop(user_data_5stars.index)

  # print(data.head(10))

  # split whole data into three folds for training, evaluating and testing
  train_data, eval_data, test_data = random_split(data, multi_ratios= ratio)

  train_data, data_info = DatasetPure.build_trainset(train_data)
  eval_data = DatasetPure.build_evalset(eval_data)
  test_data = DatasetPure.build_testset(test_data)
  tet_data = [train_data, eval_data, test_data]
  print(data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %  
  return tet_data, data_info
  #-- Libco -- 

def blog_LightGCN_data(path, data_frame, rating_threshold =4, excluded_user=None):

  #-- 유저, 아이템 매핑
  rating_data = pd.read_csv(path[0], index_col= data_frame[0])
  item_data = pd.read_csv(path[1], index_col= data_frame[1])
  data = pd.read_csv(path[0])

  print("1번째 데이터 삭제하기 전")
  print(rating_data.head(10))

  #-- 테스트를 위한 특정 유저의 가장 앞에 있는 데이터 삭제

  if excluded_user is not None:
    user_data = data[data[data_frame[0]] == excluded_user]
    user_data_5stars = user_data[user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0r_user_data[r_user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0

    r_user_data =  rating_data[rating_data.index == excluded_user].reset_index()

    if not user_data_5stars.empty:
      excluded_number = 15
      user_data_5stars = user_data_5stars.head(excluded_number)     
      r_user_data_5stars = r_user_data[r_user_data[data_frame[2]] == 5.0]
      data = data.drop(user_data_5stars.index)
      
      row_index = []
      for i in range(0,excluded_number):
        row_index.append(r_user_data_5stars.iloc[i].name)

      rating_data = rating_data.reset_index().drop(row_index[0])
      rating_data.set_index(data_frame[0], inplace=True)
      for i in range(1, len(row_index)) :
        rating_data = rating_data.reset_index().drop(row_index[i]-i)
      # rating_data = rating_data.drop(rating_data.iloc.nam)
        rating_data.set_index(data_frame[0], inplace=True)
  
  print("\n\n\n1번째 데이터 삭제하기 후")
  print(rating_data.head(10))

  # # rating_data의 컬럼 확인
  # print(rating_data.head(5))

  user_mapping = {index: i for i, index in enumerate(rating_data.index.unique())}
  item_mapping = {index: i for i, index in enumerate(item_data.index.unique())}

  #-- 매핑 데이터로 edge연결
  edge_index = None
  src = [user_mapping[index] for index in data[data_frame[0]]]
  dst = [item_mapping[index] for index in data[data_frame[1]]]
  edge_attr = torch.from_numpy(data[data_frame[2]].values).view(-1, 1).to(torch.long) >= rating_threshold
  
  edge_index = [[], []]
  for i in range(edge_attr.shape[0]):
    if edge_attr[i]:
        edge_index[0].append(src[i])
        edge_index[1].append(dst[i])
  edge_index = torch.tensor(edge_index)

  #-- 데이터 분할 8:1:1
  num_users, num_movies = len(user_mapping), len(item_mapping)
  num_interactions = edge_index.shape[1]
  all_indices = [i for i in range(num_interactions)]

  train_indices, test_indices = train_test_split(
      all_indices, test_size=0.2, random_state=1)
  val_indices, test_indices = train_test_split(
      test_indices, test_size=0.5, random_state=1)

  train_edge_index = edge_index[:, train_indices]
  val_edge_index = edge_index[:, val_indices]
  test_edge_index = edge_index[:, test_indices]

  print("Train Edge Data : ", train_edge_index.shape)
  print("Val Edge Data :", val_edge_index.shape)
  print("Test Edge Data : ", test_edge_index.shape)    

  #-- 연결된 엣지를 희소행렬로 변환
  # convert edge indices into Sparse Tensors: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
  from torch_geometric.utils import train_test_split_edges

  train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
      num_users + num_movies, num_users + num_movies))
  val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
      num_users + num_movies, num_users + num_movies))
  test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
      num_users + num_movies, num_users + num_movies))

  tet_edge_data = [edge_index, train_edge_index,val_edge_index, test_edge_index]  
  tet_sparse_data = [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index]
  return tet_edge_data, tet_sparse_data, user_mapping, item_mapping
#-- Blog --

#-- 모델 정의

def libco_LightGCN(data_info, embed_size =32, n_epochs = 20):
  lightgcn = LightGCN(
      task="ranking",
      data_info= data_info,
      loss_type="bpr",
      embed_size= embed_size,
      n_epochs=n_epochs,
      lr=1e-3,
      batch_size=2048,
      num_neg=1,
      device="cuda",
  )
  return lightgcn
#-- Libco --#

class Blog_LightGCN(MessagePassing):
      def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
          super().__init__()
          self.num_users, self.num_items = num_users, num_items
          self.embedding_dim, self.K = embedding_dim, K
          self.add_self_loops = add_self_loops
          self.users_emb = nn.Embedding(
              num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
          self.items_emb = nn.Embedding(
              num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0
          nn.init.normal_(self.users_emb.weight, std=0.1)
          nn.init.normal_(self.items_emb.weight, std=0.1)
      def forward(self, edge_index: SparseTensor):
          # compute \tilde{A}: symmetrically normalized adjacency matrix
          edge_index_norm = gcn_norm(
              edge_index, add_self_loops=self.add_self_loops)
          emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
          embs = [emb_0]
          emb_k = emb_0

          # multi-scale diffusion
          for i in range(self.K):
              emb_k = self.propagate(edge_index_norm, x=emb_k)
              embs.append(emb_k)
          embs = torch.stack(embs, dim=1)
          emb_final = torch.mean(embs, dim=1) # E^K
          users_emb_final, items_emb_final = torch.split(
              emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K
          # returns e_u^K, e_u^0, e_i^K, e_i^0
          return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

      def message(self, x_j: Tensor) -> Tensor:
          return x_j

      def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
          # computes \tilde{A} @ x
          return matmul(adj_t, x)

#-- 손실함수 bpr
def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2)) # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss
    return loss

def blog_LightGCN(num_users, num_movies, embed_size):
  # defines LightGCN model
  model = Blog_LightGCN(num_users, num_movies, embedding_dim = embed_size)
  return model

#-- Blog --#

#-- 모델 학습
def libco_fit(model, train_data, eval_data):
  model.fit(
        train_data,
        neg_sampling=True,
        verbose=1,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )
#-- Libco--#

# function which random samples a mini-batch of positive and negative samples
def sample_mini_batch(batch_size, edge_index):
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices

# helper function to get N_u
def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, k):
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg
# wrapper function to evaluate model
def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
        neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg
def blog_fit(model, blog_tet_edge_data, blog_tet_sparse_data, epochs = 50):
  # define contants
  ITERATIONS = epochs
  BATCH_SIZE = 2048
  LR = 1e-3
  ITERS_PER_EVAL = 300
  ITERS_PER_LR_DECAY = 300
  K = 10
  LAMBDA = 1e-6

  model = model.to(device)
  model.train()

  optimizer = optim.Adam(model.parameters(), lr=LR)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

  edge_index = blog_tet_edge_data[0].to(device)
  train_edge_index = blog_tet_edge_data[1].to(device)
  train_sparse_edge_index = blog_tet_sparse_data[0].to(device)

  val_edge_index = blog_tet_edge_data[2].to(device)
  val_sparse_edge_index = blog_tet_sparse_data[1].to(device)

  # training loop
  train_losses = []
  val_losses = []

  for iter in range(ITERATIONS):
      # forward propagation
      users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
          train_sparse_edge_index)

      # mini batching
      user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
          BATCH_SIZE, train_edge_index)
      user_indices, pos_item_indices, neg_item_indices = user_indices.to(
          device), pos_item_indices.to(device), neg_item_indices.to(device)
      users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
      pos_items_emb_final, pos_items_emb_0 = items_emb_final[
          pos_item_indices], items_emb_0[pos_item_indices]
      neg_items_emb_final, neg_items_emb_0 = items_emb_final[
          neg_item_indices], items_emb_0[neg_item_indices]

      # loss computation
      train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                            pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()

      if iter % ITERS_PER_EVAL == 0:
          model.eval()
          val_loss, recall, precision, ndcg = evaluation(
              model, val_edge_index, val_sparse_edge_index, [train_edge_index], K, LAMBDA)
          print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
          train_losses.append(train_loss.item())
          val_losses.append(val_loss)
          model.train()

      if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
          scheduler.step()
  iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]

#-- 모델 평가

def libco_eval(model, test_data):
  result = evaluate(
      model=model,
      data=test_data,
      neg_sampling=True,
      metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
  )
  print(result)

#-- Libco --

def blog_eval(model, blog_tet_edge_data, blog_tet_sparse_data , K = 10):
  # evaluate on test set
  LAMBDA = 1e-6
  model.eval()
  test_edge_index = blog_tet_edge_data[3].to(device)
  test_sparse_edge_index = blog_tet_sparse_data[2].to(device)

  test_loss, test_recall, test_precision, test_ndcg = evaluation(
              model, test_edge_index, test_sparse_edge_index, [blog_tet_edge_data[1], blog_tet_edge_data[2]], K, LAMBDA)

  print(f"[test_loss: {round(test_loss, 5)}, test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}, test_ndcg@{K}: {round(test_ndcg, 5)}")
#-- Blog --

#-- 모델 예측

def lilco_predict(model, user_id, path, K = 10):
  df = pd.read_csv(path[1])
  rec_arr = model.recommend_user(user=user_id, n_rec=K).get(user_id)
  
  titles = []
  genres = []
  for i in rec_arr:
    title = df[df['movieId'] == i]['title']
    genre = df[df['movieId'] == i]['genres']
    titles.append(title)
    genres.append(genre)
    # print(f"title: {title}, genres: {genre} ")

  return titles, genres
#-- Libco  --#

def blog_predict(model, user_id ,TopK, path, user_mapping, item_mapping, edge_index):

  df = pd.read_csv(path[1])
  movieid_title = pd.Series(df.title.values,index=df.movieId).to_dict()
  movieid_genres = pd.Series(df.genres.values,index=df.movieId).to_dict()

  user_pos_items = get_user_positive_items(edge_index)
  user = user_mapping[user_id]
  e_u = model.users_emb.weight[user]
  scores = model.items_emb.weight @ e_u

  values, indices = torch.topk(scores, k=len(user_pos_items[user]) + TopK)

  movies = [index.cpu().item() for index in indices if index in user_pos_items[user]][:TopK]
  movie_ids = [list(item_mapping.keys())[list(item_mapping.values()).index(movie)] for movie in movies]
  titles = [movieid_title[id] for id in movie_ids]
  genres = [movieid_genres[id] for id in movie_ids]
 
  # print(f"Here are some movies that user {user_id} rated highly")
  # for i in range(TopK):
  #     print(f"title: {titles[i]}, genres: {genres[i]} ")

  movies = [index.cpu().item() for index in indices if index not in user_pos_items[user]][:TopK]
  movie_ids = [list(item_mapping.keys())[list(item_mapping.values()).index(movie)] for movie in movies]
  titles = [movieid_title[id] for id in movie_ids]
  genres = [movieid_genres[id] for id in movie_ids]

  # print(f"Here are some suggested movies for user {user_id}")
  return titles, genres

#-- 최종 함수

def libco(data_frame, path, embed_size, n_epochs, excluded_user):
  lib_tet_data, lib_data_info = libco_LightGCN_data(path[0], data_frame, excluded_user = excluded_user_id)
  lib_model = libco_LightGCN(lib_data_info, embed_size =embed_size, n_epochs = int(n_epochs/3)) 
  libco_fit(lib_model, lib_tet_data[0], lib_tet_data[1])
  libco_eval(lib_model, lib_tet_data[2])
  titles, genres = lilco_predict(lib_model,user_id, path, K = K)
  return titles,genres
  
def blog(data_frame, path, embed_size, n_epochs, excluded_user):
  blog_tet_edge_data, blog_tet_sparse_data, user_mapping, item_mapping = blog_LightGCN_data(path, data_frame, 4, excluded_user_id)
  blog_model = blog_LightGCN(len(user_mapping), len(item_mapping), embed_size= 32)
  blog_fit(blog_model, blog_tet_edge_data, blog_tet_sparse_data, epochs =n_epochs)
  blog_eval(blog_model, blog_tet_edge_data, blog_tet_sparse_data, 10)
  titles, genres = blog_predict(blog_model, user_id , K, path, user_mapping, item_mapping, blog_tet_edge_data[0])
  return titles,genres

#-- 원하는 파라메터 수정
data_frame = ["userId", "movieId", "rating"]
path = [rating_path, item_path]
embed_size = 16
n_epochs = 20000
user_id = 1
excluded_user = 1
K = 15

#-- 데이터 전처리, 모델정의, 모델학습, 모델평가, 모델 예측 진행
lib_titles, lib_genres = libco(data_frame, path,embed_size, n_epochs, excluded_user)
blog_titles, blog_genres = blog(data_frame, path, embed_size, n_epochs, excluded_user)

#-- 각각의 모델의 추천 데이터 확인
for i in range(K):
      print(f"title: {lib_titles[i].values}, genres: {lib_genres[i].values} ")
print("========================================================")
for i in range(K):
      print(f"title: {blog_titles[i]}, genres: {blog_genres[i]} ")

#-- 삭제한 데이터 확인
data = pd.read_csv(path[0])
user_data = data[data[data_frame[0]] == excluded_user]
user_data_5stars = user_data[user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0r_user_data[r_user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0
# print(user_data_5stars.head(10))
data = user_data_5stars.head(15)
moviedIds = data['movieId'].values

df = pd.read_csv(path[1])

titles = []
genres = []
for i in moviedIds:
    title = df[df['movieId'] == i]['title']
    genre = df[df['movieId'] == i]['genres']
    titles.append(title)
    genres.append(genre)
for i in range(len(moviedIds)):
  print(f"title: {titles[i].values}, genres: {genres[i].values} ")

```

</div>
</details>


<aside>
❗ **만약  코드를 다시 돌릴 시 항상 아래의 데이터 전처리 함수 코드를 실행 필수!**

</aside>

### **각각의 함수들 설명**

- libco_LightGCN_data : 라이브러리 GCN을 위한 데이터를 전처리하는 함수
- blog_LightGCN_data : 블로그에서 구현한 GCN을 위한 데이터를 전처리하는 함수

      **excluded_number = 15는 유저가 높은 점수를 준 아이템을 삭제하는 개수입니다.**

<aside>
❗ **만약  코드를 다시 돌릴 시 항상 아래의 해당 코드를 다시 돌려줘야 함**

</aside>

```python
def libco_LightGCN_data(path, data_frame, ratio = [0.8, 0.1, 0.1], excluded_user=None):
  #-- data
  data = pd.read_csv(path)
  data.rename(columns = {data_frame[0] : "user", data_frame[1]:"item", data_frame[2]:"label",},
              inplace = True)
  
  # print(data.head(10))
  #-- 테스트를 위한 특정 유저의 가장 앞에 있는 데이터 n개 삭제
  if excluded_user is not None:
    user_data = data[data["user"] == excluded_user]
    user_data_5stars = user_data[user_data["label"] == 5.0]  # Filter the data with a rating of 5.0
    if not user_data_5stars.empty:
      excluded_number = 15
      user_data_5stars = user_data_5stars.head(excluded_number)
      data = data.drop(user_data_5stars.index)

  # print(data.head(10))

  # split whole data into three folds for training, evaluating and testing
  train_data, eval_data, test_data = random_split(data, multi_ratios= ratio)

  train_data, data_info = DatasetPure.build_trainset(train_data)
  eval_data = DatasetPure.build_evalset(eval_data)
  test_data = DatasetPure.build_testset(test_data)
  tet_data = [train_data, eval_data, test_data]
  print(data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %  
  return tet_data, data_info
  #-- Libco -- 

def blog_LightGCN_data(path, data_frame, rating_threshold =4, excluded_user=None):

  #-- 유저, 아이템 매핑
  rating_data = pd.read_csv(path[0], index_col= data_frame[0])
  item_data = pd.read_csv(path[1], index_col= data_frame[1])
  data = pd.read_csv(path[0])

  print("1번째 데이터 삭제하기 전")
  print(rating_data.head(10))

  #-- 테스트를 위한 특정 유저의 가장 앞에 있는 데이터 삭제

  if excluded_user is not None:
    user_data = data[data[data_frame[0]] == excluded_user]
    user_data_5stars = user_data[user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0r_user_data[r_user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0

    r_user_data =  rating_data[rating_data.index == excluded_user].reset_index()

    if not user_data_5stars.empty:
      excluded_number = 15
      user_data_5stars = user_data_5stars.head(excluded_number)     
      r_user_data_5stars = r_user_data[r_user_data[data_frame[2]] == 5.0]
      data = data.drop(user_data_5stars.index)
      
      row_index = []
      for i in range(0,excluded_number):
        row_index.append(r_user_data_5stars.iloc[i].name)

      rating_data = rating_data.reset_index().drop(row_index[0])
      rating_data.set_index(data_frame[0], inplace=True)
      for i in range(1, len(row_index)) :
        rating_data = rating_data.reset_index().drop(row_index[i]-i)
      # rating_data = rating_data.drop(rating_data.iloc.nam)
        rating_data.set_index(data_frame[0], inplace=True)
  
  print("\n\n\n1번째 데이터 삭제하기 후")
  print(rating_data.head(10))

  # # rating_data의 컬럼 확인
  # print(rating_data.head(5))

  user_mapping = {index: i for i, index in enumerate(rating_data.index.unique())}
  item_mapping = {index: i for i, index in enumerate(item_data.index.unique())}

  #-- 매핑 데이터로 edge연결
  edge_index = None
  src = [user_mapping[index] for index in data[data_frame[0]]]
  dst = [item_mapping[index] for index in data[data_frame[1]]]
  edge_attr = torch.from_numpy(data[data_frame[2]].values).view(-1, 1).to(torch.long) >= rating_threshold
  
  edge_index = [[], []]
  for i in range(edge_attr.shape[0]):
    if edge_attr[i]:
        edge_index[0].append(src[i])
        edge_index[1].append(dst[i])
  edge_index = torch.tensor(edge_index)

  #-- 데이터 분할 8:1:1
  num_users, num_movies = len(user_mapping), len(item_mapping)
  num_interactions = edge_index.shape[1]
  all_indices = [i for i in range(num_interactions)]

  train_indices, test_indices = train_test_split(
      all_indices, test_size=0.2, random_state=1)
  val_indices, test_indices = train_test_split(
      test_indices, test_size=0.5, random_state=1)

  train_edge_index = edge_index[:, train_indices]
  val_edge_index = edge_index[:, val_indices]
  test_edge_index = edge_index[:, test_indices]

  print("Train Edge Data : ", train_edge_index.shape)
  print("Val Edge Data :", val_edge_index.shape)
  print("Test Edge Data : ", test_edge_index.shape)    

  #-- 연결된 엣지를 희소행렬로 변환
  # convert edge indices into Sparse Tensors: https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
  from torch_geometric.utils import train_test_split_edges

  train_sparse_edge_index = SparseTensor(row=train_edge_index[0], col=train_edge_index[1], sparse_sizes=(
      num_users + num_movies, num_users + num_movies))
  val_sparse_edge_index = SparseTensor(row=val_edge_index[0], col=val_edge_index[1], sparse_sizes=(
      num_users + num_movies, num_users + num_movies))
  test_sparse_edge_index = SparseTensor(row=test_edge_index[0], col=test_edge_index[1], sparse_sizes=(
      num_users + num_movies, num_users + num_movies))

  tet_edge_data = [edge_index, train_edge_index,val_edge_index, test_edge_index]  
  tet_sparse_data = [train_sparse_edge_index, val_sparse_edge_index, test_sparse_edge_index]
  return tet_edge_data, tet_sparse_data, user_mapping, item_mapping
#-- Blog --
```

1. **모델정의**

**2개 모델 모두 기본값으로 진행하며 embed_size와 epochs만 인자로 넘겨줍니다.**

- libco_LightGCN : 라이브러리 모델 정의
- Blog_LightGCN : 블로그 모델 정의

```python
def libco_LightGCN(data_info, embed_size =32, n_epochs = 20):
  lightgcn = LightGCN(
      task="ranking",
      data_info= data_info,
      loss_type="bpr",
      embed_size= embed_size,
      n_epochs=n_epochs,
      lr=1e-3,
      batch_size=2048,
      num_neg=1,
      device="cuda",
  )
  return lightgcn
#-- Libco --#

class Blog_LightGCN(MessagePassing):
      def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
          super().__init__()
          self.num_users, self.num_items = num_users, num_items
          self.embedding_dim, self.K = embedding_dim, K
          self.add_self_loops = add_self_loops
          self.users_emb = nn.Embedding(
              num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
          self.items_emb = nn.Embedding(
              num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0
          nn.init.normal_(self.users_emb.weight, std=0.1)
          nn.init.normal_(self.items_emb.weight, std=0.1)
      def forward(self, edge_index: SparseTensor):
          # compute \tilde{A}: symmetrically normalized adjacency matrix
          edge_index_norm = gcn_norm(
              edge_index, add_self_loops=self.add_self_loops)
          emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
          embs = [emb_0]
          emb_k = emb_0

          # multi-scale diffusion
          for i in range(self.K):
              emb_k = self.propagate(edge_index_norm, x=emb_k)
              embs.append(emb_k)
          embs = torch.stack(embs, dim=1)
          emb_final = torch.mean(embs, dim=1) # E^K
          users_emb_final, items_emb_final = torch.split(
              emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K
          # returns e_u^K, e_u^0, e_i^K, e_i^0
          return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

      def message(self, x_j: Tensor) -> Tensor:
          return x_j

      def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
          # computes \tilde{A} @ x
          return matmul(adj_t, x)

#-- 손실함수 bpr
def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, lambda_val):
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2)) # L2 loss

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss
    return loss

def blog_LightGCN(num_users, num_movies, embed_size):
  # defines LightGCN model
  model = Blog_LightGCN(num_users, num_movies, embedding_dim = embed_size)
  return model

#-- Blog --#
```

1. 모델학습 
- libco_fit : 라이브러리 모델 학습
- blog_fit : 위 함수를 제외하고는 모두 블로그 모델을 위한 함수들입니다.

```python
def libco_fit(model, train_data, eval_data):
  model.fit(
        train_data,
        neg_sampling=True,
        verbose=1,
        eval_data=eval_data,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )
#-- Libco--#

# function which random samples a mini-batch of positive and negative samples
def sample_mini_batch(batch_size, edge_index):
    edges = structured_negative_sampling(edge_index)
    edges = torch.stack(edges, dim=0)
    indices = random.choices(
        [i for i in range(edges[0].shape[0])], k=batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]
    return user_indices, pos_item_indices, neg_item_indices

# helper function to get N_u
def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items

# computes recall@K and precision@K
def RecallPrecision_ATk(groundTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)  # number of correctly predicted items per user
    # number of items liked by each user in the test set
    user_num_liked = torch.Tensor([len(groundTruth[i])
                                  for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    return recall.item(), precision.item()

# computes NDCG@K
def NDCGatK_r(groundTruth, r, k):
    assert len(r) == len(groundTruth)

    test_matrix = torch.zeros((len(r), k))

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

# wrapper function to get evaluation metrics
def get_metrics(model, edge_index, exclude_edge_indices, k):
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    # get ratings between every user and item - shape is num users x num movies
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        # gets all the positive items for each user from the edge index
        user_pos_items = get_user_positive_items(exclude_edge_index)
        # get coordinates of all edges to exclude
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        # set ratings of excluded edges to large negative value
        rating[exclude_users, exclude_items] = -(1 << 10)

    # get the top k recommended items for each user
    _, top_K_items = torch.topk(rating, k=k)

    # get all unique users in evaluated split
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)

    # convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    # determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, ndcg
# wrapper function to evaluate model
def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val):
    # get embeddings
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
        sparse_edge_index)
    edges = structured_negative_sampling(
        edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[
        pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[
        neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0,
                    neg_items_emb_final, neg_items_emb_0, lambda_val).item()

    recall, precision, ndcg = get_metrics(
        model, edge_index, exclude_edge_indices, k)

    return loss, recall, precision, ndcg
def blog_fit(model, blog_tet_edge_data, blog_tet_sparse_data, epochs = 50):
  # define contants
  ITERATIONS = epochs
  BATCH_SIZE = 2048
  LR = 1e-3
  ITERS_PER_EVAL = 300
  ITERS_PER_LR_DECAY = 300
  K = 10
  LAMBDA = 1e-6

  model = model.to(device)
  model.train()

  optimizer = optim.Adam(model.parameters(), lr=LR)
  scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

  edge_index = blog_tet_edge_data[0].to(device)
  train_edge_index = blog_tet_edge_data[1].to(device)
  train_sparse_edge_index = blog_tet_sparse_data[0].to(device)

  val_edge_index = blog_tet_edge_data[2].to(device)
  val_sparse_edge_index = blog_tet_sparse_data[1].to(device)

  # training loop
  train_losses = []
  val_losses = []

  for iter in range(ITERATIONS):
      # forward propagation
      users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
          train_sparse_edge_index)

      # mini batching
      user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(
          BATCH_SIZE, train_edge_index)
      user_indices, pos_item_indices, neg_item_indices = user_indices.to(
          device), pos_item_indices.to(device), neg_item_indices.to(device)
      users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
      pos_items_emb_final, pos_items_emb_0 = items_emb_final[
          pos_item_indices], items_emb_0[pos_item_indices]
      neg_items_emb_final, neg_items_emb_0 = items_emb_final[
          neg_item_indices], items_emb_0[neg_item_indices]

      # loss computation
      train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,
                            pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()

      if iter % ITERS_PER_EVAL == 0:
          model.eval()
          val_loss, recall, precision, ndcg = evaluation(
              model, val_edge_index, val_sparse_edge_index, [train_edge_index], K, LAMBDA)
          print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss, 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
          train_losses.append(train_loss.item())
          val_losses.append(val_loss)
          model.train()

      if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
          scheduler.step()
  iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
  plt.plot(iters, train_losses, label='train')
  plt.plot(iters, val_losses, label='validation')
  plt.xlabel('iteration')
  plt.ylabel('loss')
  plt.title('training and validation loss curves')
  plt.legend()
  plt.show()
```

1. 모델 평가

모델 각각에 대해서 평가

```python
def libco_eval(model, test_data):
  result = evaluate(
      model=model,
      data=test_data,
      neg_sampling=True,
      metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
  )
  print(result)

#-- Libco --

def blog_eval(model, blog_tet_edge_data, blog_tet_sparse_data , K = 10):
  # evaluate on test set
  LAMBDA = 1e-6
  model.eval()
  test_edge_index = blog_tet_edge_data[3].to(device)
  test_sparse_edge_index = blog_tet_sparse_data[2].to(device)

  test_loss, test_recall, test_precision, test_ndcg = evaluation(
              model, test_edge_index, test_sparse_edge_index, [blog_tet_edge_data[1], blog_tet_edge_data[2]], K, LAMBDA)

  print(f"[test_loss: {round(test_loss, 5)}, test_recall@{K}: {round(test_recall, 5)}, test_precision@{K}: {round(test_precision, 5)}, test_ndcg@{K}: {round(test_ndcg, 5)}")
```

1. 예측
- lilco_predict
- blog_predict

2개의 해당 user에게  K개의 영화 제목과 장르를 추천하여  return 합니다.

```python
def lilco_predict(model, user_id, path, K = 10):
  df = pd.read_csv(path[1])
  rec_arr = model.recommend_user(user=user_id, n_rec=K).get(user_id)
  
  titles = []
  genres = []
  for i in rec_arr:
    title = df[df['movieId'] == i]['title']
    genre = df[df['movieId'] == i]['genres']
    titles.append(title)
    genres.append(genre)
    # print(f"title: {title}, genres: {genre} ")

  return titles, genres
#-- Libco  --#

def blog_predict(model, user_id ,TopK, path, user_mapping, item_mapping, edge_index):

  df = pd.read_csv(path[1])
  movieid_title = pd.Series(df.title.values,index=df.movieId).to_dict()
  movieid_genres = pd.Series(df.genres.values,index=df.movieId).to_dict()

  user_pos_items = get_user_positive_items(edge_index)
  user = user_mapping[user_id]
  e_u = model.users_emb.weight[user]
  scores = model.items_emb.weight @ e_u

  values, indices = torch.topk(scores, k=len(user_pos_items[user]) + TopK)

  movies = [index.cpu().item() for index in indices if index in user_pos_items[user]][:TopK]
  movie_ids = [list(item_mapping.keys())[list(item_mapping.values()).index(movie)] for movie in movies]
  titles = [movieid_title[id] for id in movie_ids]
  genres = [movieid_genres[id] for id in movie_ids]
 
  # print(f"Here are some movies that user {user_id} rated highly")
  # for i in range(TopK):
  #     print(f"title: {titles[i]}, genres: {genres[i]} ")

  movies = [index.cpu().item() for index in indices if index not in user_pos_items[user]][:TopK]
  movie_ids = [list(item_mapping.keys())[list(item_mapping.values()).index(movie)] for movie in movies]
  titles = [movieid_title[id] for id in movie_ids]
  genres = [movieid_genres[id] for id in movie_ids]

  # print(f"Here are some suggested movies for user {user_id}")
  return titles, genres
```

위 함수들을 모두 하나의 함수로 만들고 예측된 K개의 영화정보들을 반환해준다.

```python
def libco(data_frame, path, embed_size, n_epochs, excluded_user):
  lib_tet_data, lib_data_info = libco_LightGCN_data(path[0], data_frame, excluded_user = excluded_user_id)
  lib_model = libco_LightGCN(lib_data_info, embed_size =embed_size, n_epochs = int(n_epochs/3)) 
  libco_fit(lib_model, lib_tet_data[0], lib_tet_data[1])
  libco_eval(lib_model, lib_tet_data[2])
  titles, genres = lilco_predict(lib_model,user_id, path, K = K)
  return titles,genres
  
def blog(data_frame, path, embed_size, n_epochs, excluded_user):
  blog_tet_edge_data, blog_tet_sparse_data, user_mapping, item_mapping = blog_LightGCN_data(path, data_frame, 4, excluded_user_id)
  blog_model = blog_LightGCN(len(user_mapping), len(item_mapping), embed_size= 32)
  blog_fit(blog_model, blog_tet_edge_data, blog_tet_sparse_data, epochs =n_epochs)
  blog_eval(blog_model, blog_tet_edge_data, blog_tet_sparse_data, 10)
  titles, genres = blog_predict(blog_model, user_id , K, path, user_mapping, item_mapping, blog_tet_edge_data[0])
  return titles,genres
```

아래 코드에서 수정 가능한 변수

- embed_size
- n_epochs
- user_id : 추천해줄 유저
- excluded_user : 추천 유저와 같은 값으로 설정
- K : K개를 자유롭게 설정

```python
data_frame = ["userId", "movieId", "rating"]
path = [rating_path, item_path]
embed_size = 16
n_epochs = 20000
user_id = 1
excluded_user = 1
K = 15

lib_titles, lib_genres = libco(data_frame, path,embed_size, n_epochs, excluded_user)
blog_titles, blog_genres = blog(data_frame, path, embed_size, n_epochs, excluded_user)
```

라이브러리와 블로그 모델의 예측값들 출력

```python

for i in range(K):
      print(f"title: {lib_titles[i].values}, genres: {lib_genres[i].values} ")
print("========================================================")
for i in range(K):
      print(f"title: {blog_titles[i]}, genres: {blog_genres[i]} ")
```

내가 삭제한 데이터 확인

```python
data = pd.read_csv(path[0])
user_data = data[data[data_frame[0]] == excluded_user]
user_data_5stars = user_data[user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0r_user_data[r_user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0
# print(user_data_5stars.head(10))
data = user_data_5stars.head(15)
moviedIds = data['movieId'].values

df = pd.read_csv(path[1])

titles = []
genres = []
for i in moviedIds:
    title = df[df['movieId'] == i]['title']
    genre = df[df['movieId'] == i]['genres']
    titles.append(title)
    genres.append(genre)
for i in range(len(moviedIds)):
  print(f"title: {titles[i].values}, genres: {genres[i].values} ")
```



### LightGCN

해당 모델이 실제로 평가를 제대로 하는지 확인하기 위해서 user가 가장 높게 평가한 데이터 1개를 제외하고 학습 후 해당 아이템을 추천해주는지 확인

1. 데이터 프레임에서 해당 유저가 5점을 평가한 데이터 중 15개를 제거

```python
#-- 테스트를 위한 특정 유저의 가장 앞에 있는 데이터 n개 삭제
  if excluded_user is not None:
    user_data = data[data["user"] == excluded_user]
    user_data_5stars = user_data[user_data["label"] == 5.0]  # Filter the data with a rating of 5.0
    if not user_data_5stars.empty:
      excluded_number = 15
      user_data_5stars = user_data_5stars.head(excluded_number)
      data = data.drop(user_data_5stars.index)

if excluded_user is not None:
    user_data = data[data[data_frame[0]] == excluded_user]
    user_data_5stars = user_data[user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0r_user_data[r_user_data[data_frame[2]] == 5.0]  # Filter the data with a rating of 5.0

    r_user_data =  rating_data[rating_data.index == excluded_user].reset_index()

    if not user_data_5stars.empty:
      excluded_number = 3
      user_data_5stars = user_data_5stars.head(excluded_number)     
      r_user_data_5stars = r_user_data[r_user_data[data_frame[2]] == 5.0]
      data = data.drop(user_data_5stars.index)
      row_index = []
      for i in range(0,excluded_number):
        row_index.append(r_user_data_5stars.iloc[i].name)

      rating_data = rating_data.reset_index().drop(row_index[0])
      rating_data.set_index(data_frame[0], inplace=True)
      for i in range(1, len(row_index)) :
        rating_data = rating_data.reset_index().drop(row_index[i]-i)
      # rating_data = rating_data.drop(rating_data.iloc.nam)
        rating_data.set_index(data_frame[0], inplace=True)
```

위 과정을 통해 excluded_user가 가장 높게 평가한 데이터 중 가장 앞에 있는 데이터를 삭제한다.

- user1이 5점을 평가한 데이터중 가장 위에 있는 15개 데이터를 삭제 후 진행
    
    
    | userId | movieId | rating | timestamp |
    | --- | --- | --- | --- |
    | 1 | 1 | 4.0 | 964982703 |
    | 1 | 3 | 4.0 | 964981247 |
    | 1 | 6 | 4.0 | 964982224 |
    | 1 | 47 | 5.0 | 964983815 |
    | 1 | 50 | 5.0 | 964982931 |
    | 1 | 70 | 3.0 | 964982400 |
    | 1 | 101 | 5.0 | 964980868 |
    | 1 | 110 | 4.0 | 964982176 |
    | 1 | 151 | 5.0 | 964984041 |
    | 1 | 157 | 5.0 | 964984100 |
- 데이터 삭제 이후
    
    
    | userId | movieId | rating | timestamp |
    | --- | --- | --- | --- |
    | 1 | 1 | 4.0 | 964982703 |
    | 1 | 3 | 4.0 | 964981247 |
    | 1 | 6 | 4.0 | 964982224 |
    | 1 | 70 | 3.0 | 964982400 |
    | 1 | 110 | 4.0 | 964982176 |
    | 1 | 163 | 5.0 | 964983650 |
    | 1 | 216 | 5.0 | 964981208 |
    | 1 | 223 | 3.0 | 964980985 |

1. 이후 이전 코드와 마찬가지로 학습 진행
    - user_id : 테스트를 진행할 유저
    - k = 15개의 상위 목록 중 확인
    
    ```python
    data_frame = ["userId", "movieId", "rating"]
    path = [rating_path, item_path]
    embed_size = 32
    n_epochs = 10000
    user_id = 1
    excluded_user = 1
    K = 15
    
    lib_titles, lib_genres = libco(data_frame, path,embed_size, n_epochs, excluded_user)
    blog_titles, blog_genres = blog(data_frame, path, embed_size, n_epochs, excluded_user)
    ```
    
2. 결과 확인
    
    ```java
    title: ['Star Trek: The Motion Picture (1979)'], genres: ['Adventure|Sci-Fi'] 
    title: ['Jaws (1975)'], genres: ['Action|Horror'] 
    title: ['Thing, The (1982)'], genres: ['Action|Horror|Sci-Fi|Thriller'] 
    title: ['Mystery Science Theater 3000: The Movie (1996)'], genres: ['Comedy|Sci-Fi'] 
    title: ['Mars Attacks! (1996)'], genres: ['Action|Comedy|Sci-Fi'] 
    title: ['Army of Darkness (1993)'], genres: ['Action|Adventure|Comedy|Fantasy|Horror'] 
    title: ['Babe: Pig in the City (1998)'], genres: ['Adventure|Children|Drama'] 
    title: ['Arachnophobia (1990)'], genres: ['Comedy|Horror'] 
    title: ['Cool Hand Luke (1967)'], genres: ['Drama'] 
    title: ['Nutty Professor, The (1996)'], genres: ['Comedy|Fantasy|Romance|Sci-Fi'] 
    title: ['Star Trek VI: The Undiscovered Country (1991)'], genres: ['Action|Mystery|Sci-Fi'] 
    title: ["One Flew Over the Cuckoo's Nest (1975)"], genres: ['Drama'] 
    title: ['Godfather, The (1972)'], genres: ['Crime|Drama'] 
    title: ['Little Shop of Horrors (1986)'], genres: ['Comedy|Horror|Musical'] 
    title: ['Aliens (1986)'], genres: ['Action|Adventure|Horror|Sci-Fi'] 
    ========================================================
    title: Shawshank Redemption, The (1994), genres: Crime|Drama 
    title: Pulp Fiction (1994), genres: Comedy|Crime|Drama|Thriller 
    **title: Star Wars: Episode IV - A New Hope (1977), genres: Action|Adventure|Sci-Fi** 
    **title: Usual Suspects, The (1995), genres: Crime|Mystery|Thriller** 
    title: Godfather, The (1972), genres: Crime|Drama 
    title: Terminator 2: Judgment Day (1991), genres: Action|Sci-Fi 
    title: Lord of the Rings: The Return of the King, The (2003), genres: Action|Adventure|Drama|Fantasy 
    title: Lord of the Rings: The Fellowship of the Ring, The (2001), genres: Adventure|Fantasy 
    title: Lord of the Rings: The Two Towers, The (2002), genres: Adventure|Fantasy 
    **title: Seven (a.k.a. Se7en) (1995), genres: Mystery|Thriller** 
    title: Apollo 13 (1995), genres: Adventure|Drama|IMAX 
    title: Sixth Sense, The (1999), genres: Drama|Horror|Mystery 
    title: Memento (2000), genres: Mystery|Thriller 
    title: One Flew Over the Cuckoo's Nest (1975), genres: Drama 
    title: Aladdin (1992), genres: Adventure|Animation|Children|Comedy|Musical
    ```
    
    ```java
    **title: ['Seven (a.k.a. Se7en) (1995)'], genres: ['Mystery|Thriller']  ->
    title: ['Usual Suspects, The (1995)'], genres: ['Crime|Mystery|Thriller']**  ->
    title: ['Bottle Rocket (1996)'], genres: ['Adventure|Comedy|Crime|Romance'] 
    title: ['Rob Roy (1995)'], genres: ['Action|Drama|Romance|War'] 
    title: ['Canadian Bacon (1995)'], genres: ['Comedy|War'] 
    title: ['Desperado (1995)'], genres: ['Action|Romance|Western'] 
    title: ['Billy Madison (1995)'], genres: ['Comedy'] 
    title: ['Dumb & Dumber (Dumb and Dumber) (1994)'], genres: ['Adventure|Comedy'] 
    **title: ['Star Wars: Episode IV - A New Hope (1977)'], genres: ['Action|Adventure|Sci-Fi'] ->**
    title: ['Tommy Boy (1995)'], genres: ['Comedy'] 
    title: ['Jungle Book, The (1994)'], genres: ['Adventure|Children|Romance'] 
    title: ['Fugitive, The (1993)'], genres: ['Thriller'] 
    title: ["Schindler's List (1993)"], genres: ['Drama|War'] 
    title: ['Tombstone (1993)'], genres: ['Action|Drama|Western'] 
    title: ['Pinocchio (1940)'], genres: ['Animation|Children|Fantasy|Musical']
    ```
    
    블로그LightGCN같은 경우 유저가 평가한 5점 짜리 영화를 3개정도 추천해줬지만 라이브러리는 1개도 추천하지 못했다..
    
    코드 수정 방법
    
    - 추가적인 아이디어 고민
    - 유사도 계산 방법 고민
    - 16일 8시 줌 미팅
    - 결과보고서 제출 준비
        - UROP : Light GCN이 결론


### 순차적 추천

### Session-based recommendation RNN 논문

### 1. INTRODUCTION

세셔 긴반 추천은 과거 클릭을 고려하지 않는 단순한 방법을 사용한다. 추천 시스템은 사용자의 선호도에 따라 사용자에게 추천하는 데 사용된다.  순차적 데이터를 처리하는데 있어서 큰 성공을 이룬 RNN을 적용하여 처음 클릭한 항목을 초기 입력으로 간주하고 랭킹 손실 함수를 사용하여 모델을 학습시키는 방식으로 RNN을 활용할 수 있다.


### 2. Related Work

**2.1 Session-Based Recommendation:**
기존의 방법들에서는 사용자의 마지막 클릭 아이템을 기준으로 유사한 아이템을 찾는데, 이로 인해 과거의 클릭 정보가 무시되는 문제가 있다. 이를 해결하기 위해  2가지 방법이 있다.

- MDP (Markov Decision Process): MDP 방법은 아이템 간의 전이 확률을 기반으로 다음 추천을 계산하는 간단한 방법이다. 하지만 아이템 수가 많아질수록, 즉 유저가 선택할 수 있는 폭이 넓어질수록 MDP 기반의 접근 방법만으로는 상태 공간을 관리하기 어려워진다.


<details>
<summary> 예시 </summary>
<div markdown="1">  

우리는 주말에 놀러갈 수 있는 여러 가지 활동을 고려하고 있습니다. 우리의 상태(State)는 현재 날씨에 따라 "맑음", "흐림", "비"로 정의됩니다. 우리는 두 가지 행동(Action)을 선택할 수 있습니다: "공원에서 산책하기" 또는 "영화관에서 영화 보기". 각 상태와 행동에는 보상(Reward)이 있습니다. 마지막으로, 날씨는 상태 전이 확률(Transition Probability)에 의해 변경됩니다.

1. 상태(State):
    - 맑음 (Sunny)
    - 흐림 (Cloudy)
    - 비 (Rainy)
2. 행동(Action):
    - 공원에서 산책하기 (Go for a walk in the park)
    - 영화관에서 영화 보기 (Go to the movie theater)
3. 보상(Reward):
    - 공원에서 산책하기:
        - 맑음: +10
        - 흐림: +5
        - 비: -10
    - 영화관에서 영화 보기:
        - 맑음: +5
        - 흐림: +10
        - 비: +15

</div>
</details>



- GFF (General Factorization Framework): GFF는 아이템에 대해 두 종류의 잠재 벡터를 사용합니다. 하나는 아이템 자체를 나타내는 벡터이고, 다른 하나는 session으로서의 아이템을 나타내는 벡터입니다. 이렇게 하면 어떤 세션은 session으로서의 아이템들의 평균으로 표현될 수 있지만 GFF는 session 간의 순서를 고려하지 않습니다.


<details>
<summary> 예시 </summary>
<div markdown="1">   

우리는 음악 스트리밍 서비스를 운영하고 있습니다. 사용자는 세션 동안 여러 개의 곡을 재생합니다. 세션은 사용자의 한 번의 방문을 나타내며, 세션 동안 재생된 곡들은 아이템으로 간주됩니다. 우리의 목표는 현재 세션에서 다음에 재생할 아이템을 추천하는 것입니다.

1. 잠재 벡터 생성:
    - 아이템 잠재 벡터(Item Embedding): 각 아이템은 고유한 잠재 벡터를 가지며, 이 벡터는 아이템 자체의 특성을 반영합니다.
    - 세션 잠재 벡터(Session Embedding): 세션은 세션 동안 재생된 아이템의 평균 잠재 벡터로 표현됩니다. 이 벡터는 세션의 특성을 반영하며, 세션 간의 순서는 고려하지 않습니다.
2. 추천 계산:
    - 현재 세션에 대해 세션 잠재 벡터를 계산합니다.
    - 세션 잠재 벡터와 아이템 잠재 벡터 사이의 유사도를 계산하여 다음에 재생할 아이템을 추천합니다.
    - 예를 들어, 세션 잠재 벡터와 각 아이템 잠재 벡터 간의 코사인 유사도를 계산할 수 있습니다.

</div>
</details>

**2.2 Deep Learning in Recommendation:**
Restricted Boltzmann Machines (RBM)은 Collaborative Filtering 모델에서 사용자와 아이템 간의 상호작용을 기반으로 우수한 성능을 보다. 최근에는 Deep Model들이 구조화되지 않은 컨텐츠에서 특징을 추출하기 위해 사용되고 전통적인 CF 방법과 함께 사용되어 왔다.

Session-Based Recommendation에서 MDP와 GFF 방법이 과거 클릭 정보를 고려하기 위한 방법이다.

Deep Learning은 RBM과 함께 사용되며 구조화되지 않은 컨텐츠에서 특징을 추출하기 위해 사용된다.

### **3. Recommendation with RNNs**

1. 문제 정의:
    - 세션 기반 추천은 사용자의 과거 상호작용 세션 데이터를 기반으로 다음에 사용자가 상호작용할 아이템을 예측하는 문제이다.
    - 각 세션은 사용자가 연속적으로 상호작용한 아이템의 시퀀스로 표현된다.
2. 임베딩:
    - 세션의 아이템 시퀀스와 각 아이템을 임베딩하여 순환 신경망에 입력으로 사용한다.
    - 임베딩은 아이템과 세션 간의 관련성을 반영하는 벡터 표현이다.
3. 순환 신경망(RNN):
    - RNN은 시퀀스 데이터를 처리하는데 효과적인 신경망 구조이다.
    - 세션의 아이템 시퀀스를 입력으로 받아 순차적으로 처리하면서 은닉 상태(hidden state)를 업데이트한다.
    - RNN은 사용자의 상호작용 패턴을 학습하여 다음 아이템을 예측하는데 활용된다.
4. 예측:
    - 순환 신경망을 통해 학습된 모델을 사용하여 다음에 상호작용할 아이템을 예측한다.
    - 예측은 소프트맥스 함수를 통해 다음 아이템에 대한 확률 분포를 계산하는 과정이다.
    - 가장 확률이 높은 아이템을 추천으로 제시한다.

<details>
<summary> 예시 </summary>
<div markdown="1">   

우리는 온라인 음악 스트리밍 서비스를 운영하고 있습니다. 각 사용자는 세션 동안 여러 곡을 연속적으로 듣습니다. 우리의 목표는 각 사용자의 세션에서 다음에 들을 곡을 추천하는 것입니다.

1. 데이터 전처리:
    - 사용자의 상호작용 데이터는 세션으로 구성되어 있습니다. 각 세션은 사용자가 연속적으로 재생한 곡들의 시퀀스로 표현됩니다.
    - 예를 들어, 사용자 A의 세션은 [곡1, 곡2, 곡3]과 같이 구성될 수 있습니다.
    - 이러한 세션 데이터를 모델의 입력으로 사용하기 위해 각 곡을 임베딩하여 벡터 형태로 변환합니다.
2. 모델 구성:
    - RNN을 사용하여 세션의 아이템 시퀀스를 처리하는 모델을 구성합니다.
    - 임베딩된 곡 벡터를 순차적으로 입력으로 주어 RNN을 통해 시퀀스를 처리합니다.
    - RNN은 각 시간 단계에서 이전 은닉 상태와 현재 입력을 기반으로 새로운 은닉 상태를 계산합니다.
3. 학습:
    - 모델은 학습 데이터를 사용하여 파라미터를 조정합니다.
    - 각 세션의 입력 시퀀스와 실제로 다음에 재생한 곡을 비교하여 모델의 예측 오차를 최소화하도록 학습합니다.
    - 손실 함수를 사용하여 예측 오차를 측정하고, 역전파 알고리즘을 통해 모델의 파라미터를 업데이트합니다.
4. 추천:
    - 학습된 모델을 사용하여 다음에 재생할 곡을 추천합니다.
    - 사용자의 현재 세션을 입력으로 주어 모델이 다음 곡에 대한 확률 분포를 계산합니다.
    - 가장 확률이 높은 곡을 추천으로 제시합니다.

예를 들어, 사용자 A의 현재 세션은 [곡1, 곡2]입니다. 모델은 입력으로 받은 세션을 기반으로 다음에 재생할 곡인 곡3을 추천할 수 있습니다. 이 추천은 사용자의 과거 상호작용 패턴을 학습하여 개인화된 추천을 제공하는 것입니다.

이와 같이 RNN을 사용한 세션 기반 추천은 사용자의 과거 상호작용을 기반으로 다음에 상호작용할 아이템을 예측하여 추천하는 방법입니다.

</div>
</details>