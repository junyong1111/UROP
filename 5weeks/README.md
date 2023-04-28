# 5주차 (GNN을 이용하여 추천 시스템 구현 NGCF)# 5주차 (GNN을 이용하여 추천 시스템 구현 NGCF)

<details>
<summary> 5주차 GNN(NGCF) </summary>
<div markdown="1">   

### Collaborative Filtering(협업 필터링)
<details>
<summary> Collaborative Filtering(협업 필터링)  </summary>
<div markdown="1">   

<details>
<summary> Collaborative Filtering </summary>
<div markdown="1">   

### **Collaborative Filtering(협업 필터링)**

다른 사용자의 구매 이력, 평가 등을 바탕으로 다른 사용자와의 유사도를 측정하여 사용자가 좋아할만한 상품을 추천하는 기술이다.

ex) A,B 두 명의 사용자가 있다고 가정

- A : 햄버거를 좋아함
- B :  A와 B의 음식 취향이 비슷함
- B 에게 A가 좋아하는 햄버거를 추천

### **Collaborative Filtering 종류**

- **Neighborhood Method**
    - 유저와 아이템의 데이터를 연관하여 추천하기 때문에 쉽다.
    - 복잡한 계산 과정이 필요하여 확장이 힘들다.
- **Latent Factor Model**
    - 기본적으로 사용
    - **Matrix Factorization**
        - **Netflix Prize 논문에서 사용**
    

**Matrix Factorization**

유저와 아이템간의 상호작용에는 유저의 행동, 아이템 평점에 영향을 주는 **잠재된 요인(Latent factor)**가 있고, 그 **잠재된 요인을 고려**하여 유저에게 적합한 아이템을 추천하는 방법

**큰 행렬을 작은 두 개의 행렬로 분해**하는 기법으로 데이터의 잠재적인 구조를 추출할 수 있다. 예를 들어, 사용자-아이템 행렬을 분해하면 사용자와 아이템 간의 특성을 추출할 수 있다. 이렇게 추출된 특성을 사용하여 사용자와 아이템을 새로운 공간에서 표현하면, 이를 활용하여 사용자에게 아직 구매하지 않은 상품을 추천할 수 있다. 

$유저-아이템 행렬 = 유저잠재요인 * 아이템잠재요인$

<img width="1293" alt="1" src="https://user-images.githubusercontent.com/79856225/235130054-89c01428-e5f1-478d-ae0b-7ea4293f3596.png">

<img width="414" alt="2" src="https://user-images.githubusercontent.com/79856225/235130076-4e758a12-7831-442f-9495-fe2329e0018c.png">

하지만 Matrix Factorization의 경우 내적을 사용하기 때문에  복잡한 유저와 아이템의 관계를 1차원 레이어에 담아내기 어렵다는 문제점이 발생

</div>
</details>

<details>
<summary> Neural Collaborative Filtering </summary>
<div markdown="1">   

### Neural Collaborative Filtering

**Deep learning** 모델을 이용하여 추천 시스템을 구현하는 방법이다.

NCF 모델은 사용자와 아이템 간의 상호작용 데이터를 입력으로 받아, 사용자와 아이템 간의 상호작용 예측을 위해 neural network를 사용한다. 일반적으로 NCF 모델은 사용자와 아이템을 표현하는 임베딩(Embedding) 층과, 이를 조합하여 예측하는 fully connected 레이어들로 이루어져 있다.

**NCF 모델은 deep learning을 사용하기 때문에, 기존의 MF 방법에 비해 더 복잡한 패턴을 추출할 수 있다.** 또한, 큰 규모의 데이터셋에 대한 적응력도 높습니다.

하지만, NCF 모델의 경우 **학습 시간이 오래 걸리며**,, MF 방법보다 파라미터 수가 더 많기 때문에, 적은 데이터셋에서는 **과적합 문제가 발생할 가능성**이 높습니다.

![3](https://user-images.githubusercontent.com/79856225/235130079-78e39f95-b652-43b3-9264-c1112e09048c.png)

- 사용자 서브넷
- 아이템 서브넷

각 서브넷은 사용자 및 아이템에 대한 정보를 입력받아 잠재요인(Latent factor)을 추출하는 역할은 하며, 두 개의 서브넷은 MLP로 구성되어 있다. MLP는 잠재요인을 특징으로 표현하는데 사용되며 이 특징은 각 사용자와 아이템 간의 상호작용을 예측하기 위해 내적으로 결합된다.

정리하자면 NCF는 사용자와 아이템의 특징 벡터를 입력으로 받고, 이 를 MLP에서 잠재요인으로 변환하고, 잠재요인 벡터를 내적하여 사용자와 아이템 간의 상호작용을 예측하는 것이다.

**Generalized MF(GMF)**

GMF는 Matrix Factorization (MF) 모델의 변형으로 MF 모델과는 달리 사용자와 아이템의 벡터를 요소별 곱셈으로 구한다.

내적은 요소별 곱셈으로 표현될 수 있어 GMF는 이러한 특성을 사용하여 벡터 내적 대신 요소별 곱셈을 사용하여 적은 수의 파라미터로도 MF보다 더 높은 성능을 발휘할 수 있다.

![4](https://user-images.githubusercontent.com/79856225/235130083-56c30d49-638b-4c1a-b94b-864cf0a01c1d.png)

</div>
</details>

</div>
</details>

### Neural Graph Collaborative Filtering

NGCF(Neural Graph Collaborative Filtering)는 그래프 신경망(Graph Neural Network)을 기반으로한 협업 필터링 알고리즘으로 그래프 신경망은 노드 간의 상호작용 관계를 모델링할 수 있는 강력한 모델링 도구로, 이를 활용하여 사용자와 아이템 간의 상호작용 정보를 모델링하여 추천을 한다. 잠재 요인 간의 상호작용을 고려하지 않고, **단순히 내적을 사용하여 예측을 수행하는 기존 모델들과 달리 잠재 요인 간의 상호작용을 고려하여 더 나은 성능**을 보인다.

![1](https://user-images.githubusercontent.com/79856225/235130693-462f32f7-9e5a-4a35-b47c-e2d14be820e5.png)


**High-order Connectivity for u1**

**경로 길이가 1보다 큰 노드에서 u1에 도달하는 경로를 의미하며 이를 통해 3가지를 알 수 있음.**

1. u1과 u2가 i2와 상호작용함으로 두 유저간의 **행동 유사성**이 존재
2. u1과 유사한 u2가 i4를 소비함 → **u1이 i4를 소비할 가능성**이 존재
3. u1은 i5보다 i4에 더 **큰 관심이 있을 가능성**이 존재
    1. u1은 u2와 i2와 상호작용함으로 연결되어 있으며 u2은 i4와 i5를 소비
    2. u1과 u3은 i3에 상호작용함으로 연결되어 있으며 u3은 i4를 소비

**NGCF 특징**

1. **그래프 신경망**을 활용하여 사용자와 아이템 간의 상호작용 정보를 모델링
2. 메모리 기반 협업 필터링 알고리즘보다 훨씬 빠르고 효율적으로 추천
3. 주변 이웃의 상호작용 정보를 잘 고려하여 정확한 추천
4. 다양한 협업 필터링 알고리즘과 비교하여 높은 성능

**NGCF 구조**

![2](https://user-images.githubusercontent.com/79856225/235130837-b57c7c9a-768d-4249-91e6-5d902b22672e.png)


1. **Embedding Layer**
- 유저와 아이템에 대해 임베딩 하는 레이어
- end - to end 방식으로 학습

1. **Embedding Propagation Layers**
- 논문에서 제안하는 중요한 레이어이며 기존 논문들과 차별되는 부분
- 유저와 아이템의 표현을 관련 짓기 위해 주변 유저와 아이템의 1차 연결 정보를 명시적으로 활용
- 그래프 구조를 따라 협업 시그널을 캡쳐
- 유저 및 아이템 임베딩을 개선하기 위해 GNN 메시지 전달 아키텍쳐 구축

**Embedding Propagation Layers**

연결된 유저-아이템 간의 임베딩 전파를 수행하며, **Message Construction,  Message Aggregation** 두 가지 주요작업을 진행

1. 유저가 아이템을 소비
2. 소비된 아이템 정보가 유저에게 전달
3. 특정 유저가 아이템을 소비한 정보는 아이템 feature로 사용 가능
4. 위 feature를 토대로 아이템또는 유저의 유사성을 측정

**Message Construction**

- 아이템 정보가 유저에게 전달되는 메시지를 구성
- 각각의 초기 Embedding 벡터의 값은 일반적으로 무작위로 초기화

**ex) 사용자 A와 아이템 B가 연결된 그래프가 있다고 가정**

1. 먼저, 사용자 A와 아이템 B는 각각의 Embedding 벡터를 가지고 있다. 이 Embedding 벡터는 사용자 A와 아이템 B의 속성 정보를 압축하여 나타낸 벡터
2. 사용자 A는 연결된 아이템 B와의 관계 정보를 담은 Embedding 벡터를 생성하고, 이Embedding 벡터는 아이템 B의 Embedding 벡터를 사용하여 생성
3. 아이템 B는 연결된 사용자 A와의 관계 정보를 담은 Embedding 벡터를 생성. 이 Embedding 벡터는 사용자 A의 Embedding 벡터를 사용하여 생성
4. 마지막으로, 생성된 두 개의 Embedding 벡터를 결합하여 사용자 A와 아이템 B 사이의 메시지를 생성

이러한 방식으로 각각의 노드들이 서로 메시지를 생성하고 전파하면서 그래프 전체의 정보를 반영하여 모델을 학습

**Message Aggregation**

- 유저의 표현 또는 아이템의 표현들을 집계
- 생성된 메시지를 모아서 새로운 임베딩을 생성
- 메시지를 모으는 과정에서 이웃 노드 간의 정보를 공유
- 합치거나 평균을 내는 방식

ex) 노드 u1과 이웃 노드 u2, u3, u4사이의 메시지를 모으는 과정에서 u2, u3,u4간에도 정보가 공유된다. 

<img width="610" alt="3" src="https://user-images.githubusercontent.com/79856225/235130896-a2fbe5d0-856e-4534-b845-61f229b28268.png">

이 과정에서 그래프의 연결 정도를 나타내는 라플라스 매트릭스로 Propagation Rule를 구현하면 모든 유저와 아이템에 대한 표현을 효율적으로 동시에 업데이트 가능

<img width="238" alt="4" src="https://user-images.githubusercontent.com/79856225/235130899-522a5594-8cd1-43d6-aed0-050b8a04e36a.png">

![5](https://user-images.githubusercontent.com/79856225/235130901-3649df56-3780-4ce3-a22a-451d6023c49b.png)

**3. Prediction Layer**

- 서로 다른 에이어에서 나온 표현은 각기 다른 메시지를 강조
- 유저 선호도 반영에 서로 다른 기여를 하게 됨
- 따라서 이 정보들을 모두 합쳐서 유저를 위한 최종 임베딩 구성
- 일반적으로 활성화 함수를 사용 안함
- 내적을 사용
    - 사용자의 벡터와 아이템의 벡터가 비슷할수록(상호작용이 많을수록) 예측 점수가 높아짐
    - 각각의 아이템에 대한 예측점수

4**. Out Layer**

- 내적을 사용
    - **Prediction Layer에서 나온 예측 점수로** 사용자와 아이템간의 상호작용 점수를 계산
    - 아이템 중에서 가장 선호할 만한 아이템을 최종적으로 추천
- 유저는 선호도가 가장 높은 아이템을 추천 받음

공개된 데이터를 가지고 제안한 알고리즘을 조합해



</div>
</details>

<!-- 
<details>
<summary>  </summary>
<div markdown="1">   

</div>
</details> -->