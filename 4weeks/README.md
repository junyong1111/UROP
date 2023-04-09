# 4주차(GNN을 이용하여 추천 시스템 구현 LightGCN)


<details>
<summary> GNN(Graph Neural Networks)  </summary>
<div markdown="1">   

### 1. Graph

그래프는 점(노드)과 이를 연결하는 선(엣지)의 집합이다. 노드의 이웃은 노드에 연결된 노드의 집합이다. 그래프는 방향이 있는 방향 그래프거나 방향이 없는 무방향 그래프가 존재하며, 또한 동질(모든 노드와 엣지가 동일)이거나 이질(노드 또는 엣지가 다름)일 수도 있다. 하이퍼그래프는 하나의 엣지가 두 개이상의 노드를 연결할 수 있는 그래프 유형이다. 

![1](https://user-images.githubusercontent.com/79856225/230766654-8ae81048-d2b8-42d8-a4b8-d7a3ea735628.png)

![Untitled 크게](https://user-images.githubusercontent.com/79856225/230766752-bda724d6-3934-41ec-9c3c-59f3607ce917.jpeg)


**Grpah = G(V,E)**

### 2. GNN

GNN은 Graph Neural Network의 약자로, 그래프 데이터 분석에서 사용하는 딥러닝 모델이다. 예를 들어, 소셜 네트워크는 친구 관계를 그래프로 표현할 수 있다.

GNN은 이러한 그래프 데이터를 입력으로 받아서, 각 노드의 임베딩을 추출하거나, 그래프 전체를 분류하는 등의 작업을 수행한다. GNN은 일반적으로 **메시지 전달 방식을 사용**하여, 각 노드의 **임베딩을 계산**합니다. 이때, **각 노드의 이웃 노드와의 관계를 이용**하여, 해당 노드의 임베딩을 갱신한다.

GNN은 반복적인 프로세스를 사용하여 **인접 노드의 특징 정보를 집계**하고 이를 현재 중앙 노드 표현과 통합한다. 이는 **집계 및 업데이트 작업을 모두 포함하는 여러 레이어**를 쌓아 수행된다. 그 결과 제공된 그래프 데이터를 기반으로 보다 정확한 예측이 가능하다.

**집계 단계**에서는 **평균 풀링** 또는 **어텐션 메커니**즘을 사용하여 인접 노드의 특징을 결합한다. 업데이트 단계에서는 **GRU**라는 연결 또는 합계 연산과 같은 다양한 전략을 사용하여 집계된 피처를 중앙 노드 특징과 통합한다. 

**그래프에서 중앙노드란,** 그래프 구조에서 중심에 위치한 노드를 의미한다. 예를 들어, 소셜 네트워크에서 친구 관계를 그래프로 표현했을 때, 친구들 사이에서 가장 많은 연결을 가지고 있는 노드를 중앙노드라고 부를 수 있다.

- **GNN  Task**
    - Node : 특정 노드가 어떤 레이블에 속하는지
    - Edge : 연결되지 않은 두 정점이 연결가능성이 있는지
    - Graph : 그래프 자체가 어떤 레이블에 속하는지
    ****
- **GNN의 구성 요소**

 **GNN에서 그래프를 구성하기 위해서는 3가지가 필요하다.**

1. **Vertex(node)**
2. **Edge**
3. **Feature**

<img width="944" alt="2" src="https://user-images.githubusercontent.com/79856225/230766656-19987636-5062-43af-ba06-cade7337f271.png">

**—# Vertext와 Edge와의 관계를 adjacency matrix(인접행렬)로 나타냄**

**—# Feature는 Feature metrix(특징 행렬)로 나타냄**

**Graph represctation learning이 먼저 시작하고 딥러닝을 접목시킨 것이 GNN이다.**

**추천 시스템에 사용되는 그래프 신경망(GNN)기법은 5가지가 존재한다.**

- **1. GCN(Graph Convolutional Network)**

- **2. GraphSAGE(Graph SAmple and aggreGatE)**
    
- **3. GAT(Graph Attention Network)**

- **4. GGNN(Graph Gated Neural Network)**
    
- **5. HGNN(Hypergraph Neural Network)**
    

### 3. **노드 임베딩**

**GNN은 여러 노드들의 고차원 정보를 저차원 공간으로 임베딩하여 그 특성을 파악한다.**

그래프 데이터 분석에서 노드 임베딩은 각 노드를 벡터 형태로 변환하는 것을 의미한다. 이를 통해 노드 간의 유사도를 계산하거나, 머신러닝 모델의 입력으로 활용할 수 있다.

예를 들어, 소셜 네트워크 그래프에서 각 노드는 사용자를 나타내고, 각 노드의 임베딩은 해당 **사용자의 특성을 나타내는 벡터로 변환**다. 이렇게 변환된 벡터를 이용해 **사용자 간의 유사도**를 계산하거나, 머신러닝 모델의 입력으로 활용할 수 있다. 또한, 임베딩은 다른 그래프 데이터 분석 알고리즘과 결합하여 그래프의 정보를 보다 효과적으로 활용할 수 있도록 한다.

### 3-1. 임베딩 종류

1. **그래프 스펙트럴 방법: 그래프 스펙트럴 방법은 노드 임베딩을 특잇값 분해(SVD)나 라플라스 행렬(Laplacian matrix) 등의 그래프 스펙트럴 분석을 통해 추출 이 방법은 그래프 전체의 정보를 이용해 노드 임베딩을 학습하며, 그래프의 대칭성과 성질을 이용해 임베딩을 생성한다. 대표적으로 DeepWalk, node2vec, LINE, SDNE 등이 있다.**

1. **그래프 신경망 방법: 그래프 신경망 방법은 메시지 패싱(Message Passing)을 이용해 노드의 임베딩을 학습한다. 메시지 패싱은 이웃 노드들과의 정보교환을 통해 노드의 임베딩을 계산한다. 이 방법은 노드간의 구조적인 관계와 특성을 잘 반영할 수 있으며, 그래프의 구조와 노드의 특성을 모두 고려할 수 있다. 대표적으로 GCN, GAT, GraphSAGE, Graph Convolutional Matrix Completion (GCMC) 등이 있습니다.**
- **Neighborhood-based Methods**
    
    노드의 임베딩을 학습하기 위해 **노드의 이웃 노드들의 특징을 이용하는 방법**. 이웃 노드들의 특징을 평균 내거나, 가중 평균을 취하는 등의 방법으로 노드의 임베딩을 계산한다. 예를 들어, **GCN(Graph Convolutional Network)은 이웃 노드들의 특징과 가중치를 합해서 노드의 임베딩**을 계산한다.
    
- **Random Walk-based Methods**
    
    노드의 임베딩을 학습하기 위해 **무작위로 그래프를 탐색하는 방법**입니다. 무작위로 그래프를 탐색하면서 노드의 이웃 노드들과의 유사도를 계산하고, 이를 이용해 노드의 임베딩을 학습합니다. 예를 들어, **DeepWalk는 무작위로 그래프를 탐색하**면서 각 노드와 인접한 노드들의 시퀀스를 생성하고, 이를 이용해 Word2Vec과 유사한 방법으로 노드의 임베딩을 학습합니다.
    
- **Spectral-based Methods**
    
    노드의 임베딩을 학습하기 위해 **그래프의 라플라시안 행렬을 이용**하는 방법입니다. 그래프의 라플라시안 행렬을 이용하면 그래프의 **고유벡터를 계산**할 수 있으며, 이를 이용해 노드의 임베딩을 계산합니다. 예를 들어, **GraphSAGE**는 그래프의 라플라시안 행렬을 이용해 노드의 임베딩을 계산합니다.
    
- **Attention-based Methods**
    
    노드의 임베딩을 학습하기 위해 **노드 간의 상호작용을 모델링**하는 방법입니다. 이 방법은 노드 간의 유사도를 계산하고, 이를 이용해 노드의 임베딩을 계산합니다. 예를 들어, **GAT(Graph Attention Network)은 이웃 노드들과의 상호작용을 계산하는데, 이를 위해 어텐션 메커니즘을 사용**합니다
    

**메시지 기반 노드 임베딩** 

<img width="954" alt="3" src="https://user-images.githubusercontent.com/79856225/230766659-15175c71-f97f-4204-a4be-8380ee043f96.png">

- A 노드에 특징을 업데이트 하기 위해서 자신과 연결된 이웃 노드들의 특징을 집계하여 자신의 특징을 업데이트 하는 방식으로 1번 연결된 이웃에 대한 특징만을 집계하는것으로 인공지능 One-layer와 유사하다.

<img width="776" alt="4" src="https://user-images.githubusercontent.com/79856225/230766663-c0c1aafb-a4cb-4b69-89b9-40d232d92e2a.png">

- 위와 같은 방식에서 A와 이웃된 노드의 또 다른 이웃들의 정보를 기준으로 레이어를 증가시킬 수 있다.

https://github.com/pyg-team/pytorch_geometric

### **데이터 셋**

1. Cora: 컴퓨터 과학 논문 데이터셋으로, 논문을 노드로, 각 논문의 키워드를 feature로 갖고 있다.
2. Citeseer: Cora와 비슷한 데이터셋으로, 컴퓨터 과학 분야의 논문을 노드로, 논문의 키워드를 feature로 갖고 있다.
3. Pubmed: 의학 분야의 논문 데이터셋으로, 논문을 노드로, 논문의 키워드를 feature로 갖고 있다.
4. Reddit: 소셜 미디어 사이트인 Reddit의 서브레딧 데이터셋으로, 각 포스트를 노드로, 사용자가 작성한 텍스트를 feature로 갖고 있다.
5. PPI (Protein-Protein Interaction): 단백질 상호작용 데이터셋으로, 각 단백질을 노드로, 단백질의 특징을 feature로 갖고 있다.

PyTorch Geometric에서는 TUDataset, Planetoid, Coauthor 등 다양한 데이터셋을 제공하고 있고,  DGL에서도 Reddit, Amazon 등 대표적인 GNN 데이터셋을 제공한다.

### 라이브러리

**Python에서 GNN 모델 설계를 지원하는 다양한 라이브러리들이 존재한다.**

1. PyTorch Geometric: PyTorch Geometric는 PyTorch 기반의 GNN 라이브러리로 대규모 그래프에서 GNN을 학습시키기 위한 다양한 유틸리티 함수와 함께 다양한 GNN 모델을 구현할 수 있다.
2. Deep Graph Library (DGL): DGL은 MXNet, PyTorch 및 TensorFlow에서 GNN 모델을 구현하고 학습시키기 위한 라이브러리. 여러 가지 그래프 레이아웃과 함께 다양한 GNN 모델과 레이어를 구현할 수 있다.
3. Spektral: Spektral은 Keras 기반의 GNN 라이브러리로 다양한 GNN 모델을 쉽게 구현하고 학습시킬 수 있다.
4. NetworkX: NetworkX는 Python에서 네트워크 분석을 위한 라이브러리입니다. NetworkX를 사용하여 GNN 모델을 구현하고 학습시킬 수 있다.




</div>
</details>



<details>
<summary> GCN(Graph Convolution Network)  </summary>
<div markdown="1">   


<img width="769" alt="1" src="https://user-images.githubusercontent.com/79856225/230766897-63c12a23-323b-4e4d-b180-22e513c1679f.png">

<img width="956" alt="2" src="https://user-images.githubusercontent.com/79856225/230766909-621ad552-d94b-4069-85de-42024d0635ce.png">

<img width="852" alt="3" src="https://user-images.githubusercontent.com/79856225/230766915-83914854-a906-4f31-a481-4bd6cec9bfec.png">

<img width="1099" alt="4" src="https://user-images.githubusercontent.com/79856225/230766920-ae657b22-cf52-44cb-9823-da3b4670042a.png">

H → 특징 행렬의 Coulmn을 통과

<img width="1088" alt="5" src="https://user-images.githubusercontent.com/79856225/230766925-2085f36d-dab7-46cb-b1e2-e17a2ea25351.png">

<img width="1256" alt="6" src="https://user-images.githubusercontent.com/79856225/230766926-c6156126-ed8b-4929-accb-89c8c476ccbc.png">

- A = 인접 행렬
- H = 특징 행렬
- W = 가중치 행렬

### **HW (특징 행렬 X 가중치 행렬)**

- 현재 예제에서는 가중치 행렬의 필터 갯수를 6개로 했지만 보통 16인듯

<img width="1232" alt="7" src="https://user-images.githubusercontent.com/79856225/230766928-bcc5641e-5d7d-4367-b42f-59830aea3932.png">

### A**HW (인접행렬 X 특징 행렬 X 가중치 행렬)**


<img width="1227" alt="8" src="https://user-images.githubusercontent.com/79856225/230766929-0e510137-2745-4eb7-80f6-515c2a46efaa.png">

**READOUT  - Permutation Invariance**

- 1번 그래프와 2번 그래프는 같은 그래프이지만 특징 행렬의 표현이 달라짐
- 그래프는 순서의 상관없이 같은 값이 나와야 함

<img width="1069" alt="9" src="https://user-images.githubusercontent.com/79856225/230766930-a4d5a443-cf0e-44d4-9756-85d37c101989.png">

**마지막 H(특징 행렬)에 MLP를 적용해서 벡터로 만들어서 활성화 함수를 통과** 

**Overall**

X → H(특징 행렬)

A → A(인접 행렬)

<img width="1170" alt="10" src="https://user-images.githubusercontent.com/79856225/230766931-7e869491-a76d-407e-8e3d-b98bd2ee973c.png">
### **Cora Dataset을 이용하여 GCN 논문 실습(Ensigner)**

[https://www.youtube.com/watch?v=JfBMCFVEuoM](https://www.youtube.com/watch?v=JfBMCFVEuoM)

### 1. Graph Convolution Networks(GCN)

- SEMI-SUPERVISED CLASSIFICATION WITH RAPH CONVOLUTIONAL NETWORKS
    
    [1609.02907.pdf](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0243e7f4-b4b7-4f93-8eee-764d06ff66ea/1609.02907.pdf)
    

ICLR 2017년에 나온 GCN에 대해 자세하게 다룬 첫 논문

### 2. Cora dataset

- Nodes = publications(책, 또는 논문)
    - 2708개의 publications(책, 논문)들이 있으며 7개의 라벨링이 되어 있음
- Edges = Citations(인용)
    - 각 논문당 인용에 대한 5429의 링크들이 존재
- Node Features = Word vectors
    - 하나의 노드에는 1433사이즈의 유니크한 words가 있다.

**구성**

Cora dataset은 2개의 파일로 구성되어 있다.

1. **Cora.content**

<img width="606" alt="11" src="https://user-images.githubusercontent.com/79856225/230766932-06e8f268-19a4-4b8b-aaa2-21fe8bfbbceb.png">

- Paper id : 논문의 고유 ID(논문의 이름이 아님)
- Word attributes :  (특징 벡터) 각 논문의 특징을 나타내는 키워드 값을 벡터화
- Class label : 각 논문이 어떤 주제인지 알 수 있음

1. **Cora.cites**
    
    <img width="632" alt="12" src="https://user-images.githubusercontent.com/79856225/230766934-144d17af-af75-4b64-a894-6ba834c042b1.png">
    
- ID of cited paper :  인용된 논문
- ID of citing paper :  인용하는 논문
    
    **—#**  **오른쪽 논문이 왼쪽 논문을 인용 1033 논문이 35논문을 인용**
    

### 3. 코드 실습

### **데이터 불러오기**

- **인접 행렬**
    - sparse 행렬(서로서로 연결된 비율이 매우 낮음)
    - 약 13,264개가 Not nonzero, 행렬의 사이즈는 2,709 X 2,709이다.
    - 약 0.18퍼센트만이 데이터에 값이 있음
- 특징 벡터
    - 각 노드에 해당하는 워드가 임베딩 되어 벡터 형태로 저장
- 레이블
    - 7개의 레이블을 클래스 번호로 레이블링(카테고리화)
- 학습 데이터
    - 0 ~ 139(140)개 데이터를 사용
- 검증 데이터
    - 200 ~ 499(300)개 데이터를 사용
- 테스트 데이터
    - 500 ~ 1400(1000)개를 사용

통상적으로 학습 : 7 검증 : 1 테스트 : 2 이지만 각자에 맞게 설정

### Model(Pytorch)

1. **Model 정의**
- **GraphConvolution**

```python
from torch import nn
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):
    def __init__(self, feature_numbers, out_numbers, bias=False) -> None:
        super().__init__()
        self.bias = bias
        self.w = Parameter(torch.FloatTensor(feature_numbers, out_numbers))
        if bias:
            self.b = Parameter(torch.FloatTensor(out_numbers))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        if self.bias is not False:
            self.b.data.uniform_(-stdv, stdv)
            
    def forward(self, x, adj):
		#-- input data 
		#-- X = 특징 행렬
		#-- adj = 인접 행렬(sparse matrix)
        support = torch.mm(x, self.w)
        out = torch.spmm(adj, support)
		#-- spmm(sparse mulple : sparse행렬 연산시 적은 메모리 사용 효율적으로 처리)
        if self.bias:
            out = out + self.b

        return out
```

- **NodeClassificationGCNN**

```python
class NodeClassificationGCNN(nn.Module):

    def __init__(self, feature_num, node_representation_dim, nclass, droupout=0.2, bias=False) -> None:
        super().__init__()
#-- 특징 행렬의 Column, Demention(CNN에서의 채널 수), output_dim
        self.gconv1 = GraphConvolution(feature_num, node_representation_dim, bias)
#-- 이전 레이어에서 나온 output, 최종 output class(7)
        self.gconv2 = GraphConvolution(node_representation_dim, nclass, bias)
        self.dropout = droupout

    def forward(self, x, adj):
        x = F.relu(self.gconv1(x, adj))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.gconv2(x, adj))
        return F.log_softmax(x, dim=1)
#-- log를 이용하여 -무한대 값을 출력 가능
```

1. **Model 실행**

```python
from GCNN import NodeClassificationGCNN

#-- 특징벡터의 column개수, 256(CNN에서의 필터의 개수), 클래수 개수)
model = NodeClassificationGCNN(features.shape[1], 256, np.max(labels.detach().numpy())+1)
```

- **학습**

```python
import torch.optim as optim
import torch.nn.functional as F

epochs=100
optimizer = optim.Adam(model.parameters(),lr=0.01)
train_losses=[]
val_losses=[]
train_accuracy=[]
val_accuracy=[]
for epoch in range(epochs):
    model.train()
    train_labels=labels[idx_train]
    val_labels=labels[idx_val]
    
    
    optimizer.zero_grad()
    output = model(features, adj)
    train_loss=F.nll_loss(output[idx_train],train_labels)
    train_losses.append(train_loss)
    t_a=accuracy(output[idx_train],train_labels)
    train_accuracy.append(t_a)
    print(f"Training epoch {epoch} ; accuracy: {accuracy(output[idx_train],train_labels)}; loss: {train_loss.item()}")
    train_loss.backward()
    optimizer.step()
    
    model.eval()
    output = model(features, adj)
    val_loss=F.nll_loss(output[idx_val],val_labels)
    val_losses.append(val_loss)
    v_a=accuracy(output[idx_val],val_labels)
    val_accuracy.append(v_a)
    print(f"Validation epoch {epoch} ; accuracy: {accuracy(output[idx_val],val_labels)}; loss: {val_loss.item()}")
```

</div>
</details>





<details>
<summary> LightGCN   </summary>
<div markdown="1">   

</div>
</details>



<!-- <details>
<summary> 3주차  </summary>
<div markdown="1">   

</div>
</details> -->