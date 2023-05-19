# 11주차 (커스텀 데이터를 이용하여 NGCF와 LightCGN 학습 및 비교)

|  | NGCF | Light GCN |
| --- | --- | --- |
| 장점 | - 사용자-아이템 상호작용을 고려하여 협업 필터링 성능 향상 | - 경량화된 모델 구조로 효율적인 학습과 추론 가능 |
|  | - 잠재 요인 모델과 그래프 신경망을 융합하여 성능 향상 | - 그래프 구조를 활용하여 사용자-아이템 관계 모델링 |
|  | - 깊은 상호작용을 고려하여 예측 성능 향상 | - 계산 및 메모리 효율성이 높은 대규모 데이터셋 지원 |
|  | - 스케일링에 강건한 모델 | - 설계 및 구현이 간단하여 사용자 적용 용이 |
| 단점 | - 계산 복잡성이 높아 학습에 시간이 오래 걸릴 수 있음 | - 상호작용을 잘 모델링하지 못할 수 있음 |
|  | - 메모리 사용량이 크며 메모리 효율성에 제약이 있을 수 있음 | - 추가 정보 고려가 제한적 |
|  | - 모델 설계와 튜닝이 복잡하여 구현과 관리가 어려울 수 있음 | - 복잡한 상호작용 모델링에 제한적 |

|  | NGCF | Light GCN |
| --- | --- | --- |
| 모델 구조 | 심층 신경망 구조 | 경량화된 그래프 합성곱 네트워크 |
| 그래프 활용 | 사용자-아이템 상호작용 데이터와 그래프 정보 활용 | 그래프 구조만 활용 |
| 추가 정보 활용 | 그래프 정보와 함께 사용자-아이템 상호작용 모델링 | 추가 정보 없이 그래프만 활용 |
| 계산 및 메모리 효율성 | 계산과 메모리 사용량이 큼 | 계산 및 메모리 효율성이 높음 |
| 모델 설계와 구현 | 설계와 튜닝이 복잡함 | 설계와 구현이 간단함 |
| 대규모 데이터셋 학습 | 스케일링에 비교적 강건함 | 대규모 데이터셋에서 효율적인 학습이 가능함 |
- NGCF의 추가 정보
1. 사용자 프로필 정보: 사용자의 나이, 성별, 관심사 등과 같은 프로필 정보를 활용할 수 있다. 이러한 정보는 사용자의 취향을 파악하고 유사한 사용자를 찾는 데 도움을 줄 수 있다.
2. 아이템 속성 정보: 아이템의 카테고리, 장르, 키워드 등과 같은 속성 정보를 활용할 수 있다. 이러한 정보는 아이템 간의 유사성을 계산하고 사용자의 관심사와 일치하는 아이템을 추천하는 데 활용될 수 있다.
3. 시간 정보: 사용자-아이템 상호작용이 발생한 시간 정보를 활용할 수 있다. 예를 들어, 최근 상호작용은 더 중요하게 고려할 수 있다.


## STEP 1 : 데이터세트 준비(커스텀, 예제)

> 추천시스텀에 사용하는 다양한 데이터셋 정보
[https://think-tech.tistory.com/6](https://think-tech.tistory.com/6)
>

<details>
<summary> 커스텀 데이터(책 평가 데이터)  </summary>
<div markdown="1"> 

- **커스텀 데이터**

Book-Crossing 데이터셋은 책정보(제목, 작가 연도 등), 평점, 유저 정보(유저 id, age)로 구성된 책 추천 데이터세트로 kaggle에서 다운로드 가능하다.

[Book-Crossing Dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset)

```markdown
### 책 평가 데이터 정보
- Book : 책번호(International Standard Book Number), 
					제목, 작가, 출판년도, 출판사로 구성된 271379개의 데이터

- Users :User ID, 나이로 총 2개로 구성된 278859개의 데이터

- Ratings : User ID, 책번호, 평가 점수 3개로 구성된 1149780개의 데이터
```

- **커스텀 데이터 - 전처리**
1. NGCF와 LightGCN에는 평점 데이터만 필요함으로 Ragins.csv만 가져와서 처리한다.
2. 평가 점수가 0점인 데이터는 제거하며 이로인해 인덱스가 꼬이므로 인덱스를 다시 리셋해준다.
3. Ragins에 있는 유저와 아이템에 대한 인덱스를 처음부터 재정렬해준다.
4. NGCF와 Light GCN에서 요구하는 데이터로 변경하기 위해서 user, item, label의 형태로 변경
    
    <img width="280" alt="1" src="https://github.com/junyong1111/UROP/assets/79856225/ab790920-aa24-41ee-b779-819c1398fd7c">

    
    전처리가 끝난 Ragins 데이터
    

```python
ratings = pd.read_csv("/content/Ratings.csv", sep=';',encoding="latin-1")
ratings = ratings[ratings['Rating'] != 0] # rating이 0인 것은 제거
ratings.reset_index(drop=True, inplace=True) #-- 결측으로 인한 인덱스 리셋

ratings['User-ID'] = pd.factorize(ratings['User-ID'])[0]
ratings['ISBN'] = pd.factorize(ratings['ISBN'])[0]

#-- 데이터가 너무 커서 10000개의 데이터로 자르고 테스트 진행
data = ratings
# data = ratings.truncate(after=433670/2)
data.set_axis(labels=['user','item','label'],axis=1,inplace=True)
#-- 데이터 레이블은 user, item label으로 변경해줘야 random_split 가능하다.
data.head()
```

- **커스텀 데이터 - 학습 데이터 스플릿**
1. 학습데이터, 평가데이터, 테스트 데이터를 8:1:1 비율로 random_split
2. NGCF, LightGCN을 사용하기 위해서는 라이브러리에서 요구하는 **DatasetPure** 자료구조로 변경

—# DatasetPure는 딕셔너리 자료구조의 형태이며 다음과 같은 Key, Value값이 매핑되어 있다.

 DatasetPure.**n_users** ⇒ n개의 유저 데이터

 DatasetPure.**n_items** ⇒ n개의 아이템 데이터

```python
train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])
print("train_data is : ", train_data.shape)
print("eval_data  is : ", eval_data.shape)
print("test_data  is : ", test_data.shape)
"""
train_data is :  (346936, 3)
eval_data  is :  (25774, 3)
test_data  is :  (25672, 3)
"""
train_data, data_info = DatasetPure.build_trainset(train_data)
eval_data = DatasetPure.build_evalset(eval_data)
test_data = DatasetPure.build_testset(test_data)
print(data_info)  #n_users: 68291, n_items: 158953, data density: 0.0032 %
``` 

</div>
</details> 

<details>
<summary>예제 데이터(MovieLens) </summary>
<div markdown="1"> 

- **예제 데이터**

MovieLens 데이터셋은 영화(ID, 제목, 연도, 장르 등), 평점, 유저 정보(유저 id, 성별, age)로 구성된 영화 데이터셋으로 다양한 모델에서 사용하는 대표적인 데이터셋이다. 다양한 크기의 데이터가 있지만 예제에서는 100000개의 데이터셋을 사용

[MovieLens](https://grouplens.org/datasets/movielens/)

```markdown
###  영화 평가 데이터 정보

- 유저 : UserID, Gender, Age, Occupation, Zip-code

- 아이템(영화) : MovieID, Title, (Year), Genres(19가지)

- 평점 : UserID, MovieID, Rating(1~5), TimeStam
```

- **예제 데이터 - 전처리**
1. NGCF와 LightGCN에는 평점 데이터만 필요함으로 Raging 데이터만 가져와서 처리한다.
2. NGCF와 Light GCN에서 요구하는 데이터로 변경하기 위해서 user, item, labe, time의 형태로 변경

```python
#-- movies data
movie_data = pd.read_csv("/content/mydrive/MyDrive/UROP/DataSet/sample_movielens_rating.dat", 
										sep="::",
	                  names=["user", "item", "label", "time"])
```

- **예제 데이터 - 학습 데이터 스플릿**
1. 학습데이터, 평가데이터, 테스트 데이터를 8:1:1 비율로 random_split
2. NGCF, LightGCN을 사용하기 위해서는 라이브러리에서 요구하는 **DatasetPure** 자료구조로 변경

—# DatasetPure는 딕셔너리 자료구조의 형태이며 다음과 같은 Key, Value값이 매핑되어 있다.

 DatasetPure.**n_users** ⇒ n개의 유저 데이터

 DatasetPure.**n_items** ⇒ n개의 아이템 데이터

```python
# split whole data into three folds for training, evaluating and testing
movie_train_data, movie_eval_data, movie_test_data = random_split(movie_data, multi_ratios=[0.8, 0.1, 0.1])

movie_train_data, movie_data_info = DatasetPure.build_trainset(movie_train_data)
movie_eval_data = DatasetPure.build_evalset(movie_eval_data)
movie_test_data = DatasetPure.build_testset(movie_test_data)
print(movie_data_info)  # n_users: 5894, n_items: 3253, data sparsity: 0.4172 %
```

|  | 커스텀 데이터 | 예제 데이터 |
| --- | --- | --- |
| 유저 데이터  | 68291(명) | 5894(명) |
| 아이템 데이터 | 158953(개) | 3253(개) |
| 평가한 데이터 비율 | 0.0032(%) | 0.4172(%) |

</div>
</details> 

## STEP 2 : 모델 정의(NGCF, LightGCN)

<details>
<summary>커스텀 데이터(NGCF, LightGCN) </summary>
<div markdown="1">   

### NGCF

라이브러리에서 설명하는 NGCF의 인자들의 기본값은 다음과 같다.

```python
class libreco.algorithms.NGCF(
			task, data_info, loss_type='cross_entropy', 
			embed_size=16, n_epochs=20, lr=0.001, lr_decay=False, epsilon=1e-08, 
			amsgrad=False, reg=None, batch_size=256, num_neg=1, node_dropout=0.0, 
			message_dropout=0.0, hidden_units=(64, 64, 64), margin=1.0, 
			sampler='random', seed=42, device='cuda', lower_upper_bound=None
)
```

- task : 해당 알고리즘이 수행할 Task에 대한 인자이며 **NGCF는 오직 *ranking만 가능**하다.*
- data_info : 데이터 전처리에서 만들어준 data정보이며 라이브러리에서 원하는 자료구조이다.
- **loss_type** : *{'cross_entropy', 'focal', 'bpr', 'max_margin'}, 가 있으며 기본적으로는 'cross_entropy'가  설정되어 있으며 focal은 직접 만들 손실함수이며 정확도가 높다고 함*
- **embed_size** : 임베딩 사이즈에 대한 인자이며 기본값 16이며 **32**으로 변경하여 사용
- **n_epochs** : 트레이닝 epochs이며 **50**으로 변경하여 사용
- lr : 학습률
- lr_decay : 학습률 감쇠를 사용할 지 여부이며 기본값 False로 설정
- epsilon : 아담 옵티마이저의 수치 안정성을 향상시키기 위한 값으로 기본 값 1e-08 설정
- amsgrad : Adam(Adaptive Moment Estimation)의 개선 버전 사용 여부 기본 값 False 설정
- reg : Regularization이며 None으로 설정
- **batch_size** : 학습 배치 사이즈이며 기본값 256으로 하면 너무 25 에폭 학습 3시간 이상 소요 되므로 LightCGN과 마찬가지로 **2048**로 변경하여 설정
- num_neg : 각각의 Positive smaple에 대한 Negative smaple의 개수이며 기본값 1로 설정
- node_dropout : node의 드롭아웃 설정이며 기본값 0.0을 적용하여 드롭아웃 적용 안함
- message_dropout : message의 드롭아웃 설정이며 마찬가지로 적용 안함
- hidden_units : 임베딩 전파에서 사용하는 히든 레이어의 개수이며 반드시 **튜플 형태**로 전달
- margin : 최대 마진손실에 사용되는 마진값으로 기본 값 1설정
- sampler : *random', 'unconsumed', 'popular가 있으며 기본값인  random 샘플 적용*
    - *random : 무작위 샘플링*
    - *unconsumed : 대상 사용자가 소비하지 않은 아이템을 샘플링*
    - *popular : 인기있는 아이템을 negative하게 샘플링*
- lower_upper_bound : 하한 및 상한 평가점수를 제한할 수 있음, 기본값인 제한없음 설정

```python
ngcf = NGCF(
    task="ranking", 
    #-- task = NGCF 모델은 오직 ranking task만 가능하다.
    data_info=data_info,
    #-- data_info = 위에서 얻은 데이터의 정보
    loss_type="focal",
    #-- loss_type = {'cross_entropy', 'focal', 'bpr', 'max_margin'}, default: 'cross_entropy'}
    embed_size=32,
    #-- 임베딩의 벡터 사이즈
    n_epochs=50,
    lr=1e-3,
    batch_size=2048,
    epsilon=1e-08,
    num_neg=1,
    hidden_units=(64, 64, 64),
    margin=1.0, 
    sampler='random',
    #-- sampler ({'random', 'unconsumed', 'popular'}, default: 'random')
    seed=42,
    device="cuda",
)
```

### LightGCN

라이브러리에서 설명하는 LightGCN의 인자들의 기본값은 다음과 같다.

```python
class libreco.algorithms.LightGCN(
			task, data_info, loss_type='cross_entropy', 
			embed_size=16, n_epochs=20, lr=0.001, lr_decay=False, epsilon=1e-08, 
			amsgrad=False, reg=None, batch_size=256, num_neg=1, dropout_rate=0.0, 
			n_layers = 3, margin=1.0, 
			sampler='random', seed=42, device='cuda', lower_upper_bound=None
)
```

- task : 해당 알고리즘이 수행할 Task에 대한 인자이며 ***ranking으로 설정***
- data_info : 데이터 전처리에서 만들어준 data정보이며 라이브러리에서 원하는 자료구조이다.
- loss_type : *{'cross_entropy', 'focal', 'bpr', 'max_margin'}, 가 있으며 기본적으로는 'bpr'가  설정되어 있으며 기본값 설정*
- embed_size : 임베딩 사이즈에 대한 인자이며 기본값 16 설정
- **n_epochs** : 트레이닝 epochs이며 **50**으로 변경하여 사용
- lr : 학습률
- lr_decay : 학습률 감쇠를 사용할 지 여부이며 기본값 False로 설정
- epsilon : 아담 옵티마이저의 수치 안정성을 향상시키기 위한 값으로 기본 값 1e-08 설정
- amsgrad : Adam(Adaptive Moment Estimation)의 개선 버전 사용 여부 기본 값 False 설정
- reg : Regularization이며 None으로 설정
- **batch_size** : 학습 배치 사이즈이며 기본값 256으로 하면 너무 25 에폭 학습 3시간 이상 소요 되므로 LightCGN과 마찬가지로 **2048**로 변경하여 설정
- num_neg : 각각의 Positive smaple에 대한 Negative smaple의 개수이며 기본값 1로 설정
- dropout_rate : 노드가 드롭아웃 될 확률이며 기본값 0으로 설정
- n_layers : GCN 레이어의 개수이며 기본값 3으로 설정
- margin : 최대 마진손실에 사용되는 마진값으로 기본 값 1설정
- sampler : *random', 'unconsumed', 'popular가 있으며 기본값인  random 샘플 적용*
    - *random : 무작위 샘플링*
    - *unconsumed : 대상 사용자가 소비하지 않은 아이템을 샘플링*
    - *popular : 인기있는 아이템을 negative하게 샘플링*
- lower_upper_bound : 하한 및 상한 평가점수를 제한할 수 있음, 기본값인 제한없음 설정

```python
Books_lightgcn = LightGCN(
    task="ranking",
    data_info=data_info,
    loss_type="bpr",
    embed_size=16,
    n_epochs=50,
    lr=1e-3,
    batch_size=2048,
    num_neg=1,
    device="cuda",
)
```

</div>
</details>

<details>
<summary> 예제 데이터(NGCF, LightGCN) </summary>
<div markdown="1">  

### NGCF

라이브러리에서 설명하는 NGCF의 인자들의 기본값은 다음과 같다.

```python
class libreco.algorithms.NGCF(
			task, data_info, loss_type='cross_entropy', 
			embed_size=16, n_epochs=20, lr=0.001, lr_decay=False, epsilon=1e-08, 
			amsgrad=False, reg=None, batch_size=256, num_neg=1, node_dropout=0.0, 
			message_dropout=0.0, hidden_units=(64, 64, 64), margin=1.0, 
			sampler='random', seed=42, device='cuda', lower_upper_bound=None
)
```

- task : 해당 알고리즘이 수행할 Task에 대한 인자이며 **NGCF는 오직 *ranking만 가능**하다.*
- data_info : 데이터 전처리에서 만들어준 data정보이며 라이브러리에서 원하는 자료구조이다.
- **loss_type** : *{'cross_entropy', 'focal', 'bpr', 'max_margin'}, 가 있으며 기본적으로는 'cross_entropy'가  설정되어 있으며 focal은 직접 만들 손실함수이며 정확도가 높다고 함*
- **embed_size** : 임베딩 사이즈에 대한 인자이며 기본값 16이며 32으로 변경
- **n_epochs** : 트레이닝 epochs이며 **50**으로 변경하여 사용
- lr : 학습률
- lr_decay : 학습률 감쇠를 사용할 지 여부이며 기본값 False로 설정
- epsilon : 아담 옵티마이저의 수치 안정성을 향상시키기 위한 값으로 기본 값 1e-08 설정
- amsgrad : Adam(Adaptive Moment Estimation)의 개선 버전 사용 여부 기본 값 False 설정
- reg : Regularization이며 None으로 설정
- **batch_size** : 학습 배치 사이즈이며 기본값 256으로 하면 너무 25 에폭 학습 3시간 이상 소요 되므로 LightCGN과 마찬가지로 **2048**로 변경하여 설정
- num_neg : 각각의 Positive smaple에 대한 Negative smaple의 개수이며 기본값 1로 설정
- node_dropout : node의 드롭아웃 설정이며 기본값 0.0을 적용하여 드롭아웃 적용 안함
- message_dropout : message의 드롭아웃 설정이며 마찬가지로 적용 안함
- hidden_units : 임베딩 전파에서 사용하는 히든 레이어의 개수이며 반드시 **튜플 형태**로 전달
- margin : 최대 마진손실에 사용되는 마진값으로 기본 값 1설정
- sampler : *random', 'unconsumed', 'popular가 있으며 기본값인  random 샘플 적용*
    - *random : 무작위 샘플링*
    - *unconsumed : 대상 사용자가 소비하지 않은 아이템을 샘플링*
    - *popular : 인기있는 아이템을 negative하게 샘플링*
- lower_upper_bound : 하한 및 상한 평가점수를 제한할 수 있음, 기본값인 제한없음 설정

```python
movie_ngcf = NGCF(
    task="ranking", 
    #-- task = NGCF 모델은 오직 ranking task만 가능하다.
    data_info= movie_data_info,
    #-- data_info = 위에서 얻은 데이터의 정보
    loss_type="focal",
    #-- loss_type = {'cross_entropy', 'focal', 'bpr', 'max_margin'}, default: 'cross_entropy'}
    embed_size=32,
    #-- 임베딩의 벡터 사이즈
    n_epochs=50,
    lr=1e-3,
    batch_size=2048,
    epsilon=1e-08,
    num_neg=1,
    hidden_units=(64, 64, 64),
    margin=1.0, 
    sampler='random',
    #-- sampler ({'random', 'unconsumed', 'popular'}, default: 'random')
    seed=42,
    device="cuda",
)
```

### LightGCN

라이브러리에서 설명하는 LightGCN의 인자들의 기본값은 다음과 같다.

```python
class libreco.algorithms.LightGCN(
			task, data_info, loss_type='cross_entropy', 
			embed_size=16, n_epochs=20, lr=0.001, lr_decay=False, epsilon=1e-08, 
			amsgrad=False, reg=None, batch_size=256, num_neg=1, dropout_rate=0.0, 
			n_layers = 3, margin=1.0, 
			sampler='random', seed=42, device='cuda', lower_upper_bound=None
)
```

- task : 해당 알고리즘이 수행할 Task에 대한 인자이며 ***ranking으로 설정***
- data_info : 데이터 전처리에서 만들어준 data정보이며 라이브러리에서 원하는 자료구조이다.
- loss_type : *{'cross_entropy', 'focal', 'bpr', 'max_margin'}, 가 있으며 기본적으로는 'bpr'가  설정되어 있으며 기본값 설정*
- embed_size : 임베딩 사이즈에 대한 인자이며 기본값 16 설정
- **n_epochs** : 트레이닝 epochs이며 **50**으로 변경하여 사용
- lr : 학습률
- lr_decay : 학습률 감쇠를 사용할 지 여부이며 기본값 False로 설정
- epsilon : 아담 옵티마이저의 수치 안정성을 향상시키기 위한 값으로 기본 값 1e-08 설정
- amsgrad : Adam(Adaptive Moment Estimation)의 개선 버전 사용 여부 기본 값 False 설정
- reg : Regularization이며 None으로 설정
- **batch_size** : 학습 배치 사이즈이며 기본값 256으로 하면 너무 25 에폭 학습 3시간 이상 소요 되므로 LightCGN과 마찬가지로 **2048**로 변경하여 설정
- num_neg : 각각의 Positive smaple에 대한 Negative smaple의 개수이며 기본값 1로 설정
- dropout_rate : 노드가 드롭아웃 될 확률이며 기본값 0으로 설정
- n_layers : GCN 레이어의 개수이며 기본값 3으로 설정
- margin : 최대 마진손실에 사용되는 마진값으로 기본 값 1설정
- sampler : *random', 'unconsumed', 'popular가 있으며 기본값인  random 샘플 적용*
    - *random : 무작위 샘플링*
    - *unconsumed : 대상 사용자가 소비하지 않은 아이템을 샘플링*
    - *popular : 인기있는 아이템을 negative하게 샘플링*
- lower_upper_bound : 하한 및 상한 평가점수를 제한할 수 있음, 기본값인 제한없음 설정

```python
Movies_lightgcn = LightGCN(
    task="ranking",
    data_info= movie_data_info,
    loss_type="bpr",
    embed_size=16,
    n_epochs=50,
    lr=1e-3,
    batch_size=2048,
    num_neg=1,
    device="cuda",
)
```

</div>
</details> 

## STEP 3 : 모델 학습(NGCF, LightGCN)

<details>
<summary> 커스텀 데이터(NGCF, LightGCN) </summary>
<div markdown="1">   

### NGCF

라이브러리에서 설명하는 학습 함수의 인자들의 기본값은 다음과 같다.

```python
fit(
		train_data, neg_sampling, 
		verbose=1, shuffle=True, eval_data=None, 
		metrics=None, k=10, eval_batch_size=8192, 
		eval_user_num=None, num_workers=0
)
```

- train_data : 학습 데이터
- neg_sampling : 학습을 위한 negative 샘플링 여부
- verbose : 1보다 큰 값을 넣으면 학습 도중 학습에 대한 자세한 여부가 나옴
- shuffle : 학습 데이터 셔플
- eval_data : 평가를 위한 eval data사용 여부이며 평가 데이터를 넣어줘야 함
- metrics : 평가 지표 리스트
- K : 상위 몇 개의 데이터를 추천할지
- eval_batch_size : 평가 데이터 배치 사이즈
- eval_user_num : 평가할 사용자 수,  양수로 설정하면 평가 데이터에서 무작위로 사용자를 샘플링
- num_workers : 학습 데이터 로딩에 사용할 서브 프로세스 수입니다. 0은 멀티프로세싱보다 느린 메인 프로세스에서 데이터를 로드한다는 의미이며 기본 값 0 설정

```python
ngcf.fit(
    train_data,
    neg_sampling=True,
    verbose=1,
    #-- verbose가 1보다 크면 자세한 내용이 학습 중 프린트
    eval_data=eval_data,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

### LightGCN

라이브러리에서 설명하는 LightGCN의 인자들의 기본값은 다음과 같다.

```python
fit(
		train_data, neg_sampling, verbose=1, shuffle=True, 
		eval_data=None, metrics=None, k=10, eval_batch_size=8192, 
		eval_user_num=None, num_workers=0)
```

- train_data : 학습 데이터
- neg_sampling : 학습을 위한 negative 샘플링 여부
- verbose : 1보다 큰 값을 넣으면 학습 도중 학습에 대한 자세한 여부가 나옴
- shuffle : 학습 데이터 셔플
- eval_data : 평가를 위한 eval data사용 여부이며 평가 데이터를 넣어줘야 함
- metrics : 평가 지표 리스트
- K : 상위 몇 개의 데이터를 추천할지
- eval_batch_size : 평가 데이터 배치 사이즈
- eval_user_num : 평가할 사용자 수,  양수로 설정하면 평가 데이터에서 무작위로 사용자를 샘플링
- num_workers : 학습 데이터 로딩에 사용할 서브 프로세스 수입니다. 0은 멀티프로세싱보다 느린 메인 프로세스에서 데이터를 로드한다는 의미이며 기본 값 0 설정

```python
Books_lightgcn.fit(
    train_data,
    neg_sampling=True,
    verbose=2,
    eval_data=eval_data,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

</div>
</details>

<details>
<summary>예제 데이터(NGCF, LightGCN) </summary>
<div markdown="1">   

### NGCF

라이브러리에서 설명하는 학습 함수의 인자들의 기본값은 다음과 같다.

```python
fit(
		train_data, neg_sampling, 
		verbose=1, shuffle=True, eval_data=None, 
		metrics=None, k=10, eval_batch_size=8192, 
		eval_user_num=None, num_workers=0
)
```

- train_data : 학습 데이터
- neg_sampling : 학습을 위한 negative 샘플링 여부
- verbose : 1보다 큰 값을 넣으면 학습 도중 학습에 대한 자세한 여부가 나옴
- shuffle : 학습 데이터 셔플
- eval_data : 평가를 위한 eval data사용 여부이며 평가 데이터를 넣어줘야 함
- metrics : 평가 지표 리스트
- K : 상위 몇 개의 데이터를 추천할지
- eval_batch_size : 평가 데이터 배치 사이즈
- eval_user_num : 평가할 사용자 수,  양수로 설정하면 평가 데이터에서 무작위로 사용자를 샘플링
- num_workers : 학습 데이터 로딩에 사용할 서브 프로세스 수입니다. 0은 멀티프로세싱보다 느린 메인 프로세스에서 데이터를 로드한다는 의미이며 기본 값 0 설정

```python
movie_ngcf.fit(
    movie_train_data,
    neg_sampling=True,
    verbose=1,
    #-- verbose가 1보다 크면 자세한 내용이 학습 중 프린트
    eval_data=movie_eval_data,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

### LightGCN

라이브러리에서 설명하는 LightGCN의 인자들의 기본값은 다음과 같다.

```python
fit(
		train_data, neg_sampling, verbose=1, shuffle=True, 
		eval_data=None, metrics=None, k=10, eval_batch_size=8192, 
		eval_user_num=None, num_workers=0)
```

- train_data : 학습 데이터
- neg_sampling : 학습을 위한 negative 샘플링 여부
- verbose : 1보다 큰 값을 넣으면 학습 도중 학습에 대한 자세한 여부가 나옴
- shuffle : 학습 데이터 셔플
- eval_data : 평가를 위한 eval data사용 여부이며 평가 데이터를 넣어줘야 함
- metrics : 평가 지표 리스트
- K : 상위 몇 개의 데이터를 추천할지
- eval_batch_size : 평가 데이터 배치 사이즈
- eval_user_num : 평가할 사용자 수,  양수로 설정하면 평가 데이터에서 무작위로 사용자를 샘플링
- num_workers : 학습 데이터 로딩에 사용할 서브 프로세스 수입니다. 0은 멀티프로세싱보다 느린 메인 프로세스에서 데이터를 로드한다는 의미이며 기본 값 0 설정

```python
Movies_lightgcn.fit(
    movie_train_data,
    neg_sampling=True,
    verbose=2,
    eval_data=movie_eval_data,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

</div>
</details>

## STEP 4 : 모델 평가(NGCF, LightGCN) ⇒ 해야함

<details>
<summary> 커스텀 데이터(NGCF, LightGCN) </summary>
<div markdown="1">   

### NGCF

라이브러리에서 설명하는 evaluate 함수의 인자들의 기본값은 다음과 같다.

```python
evaluate(
    model=ngcf,
    data=test_data,
    neg_sampling=True,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

- metrics
    
    - loss: 모델이 예측한 값과 실제 값의 차이를 측정하는 지표
    - ROC-AUC: 이진 분류 문제에서 많이 사용되는 평가지표
    - Precision: 모델이 추천한 상위 n개의 아이템 중 실제로 선호하는 아이템의 비율을 나타내는 지표
    - Recall: 실제 선호하는 아이템 중 모델이 추천한 상위 n개의 아이템의 비율을 나타내는 지표
    - NDCG: 추천 시스템에서 상위 n개의 추천 아이템이 실제 선호하는 아이템에 얼마나 가까운지 측정
    
    ```python
    - loss': 0.5198264936902364,
    - roc_auc': 0.8554520877989423,
    - precision': 0.003323974277692238,
    - recall': 0.015959115378586058,
    - ndcg': 0.015552468873683481}
    ```
    

**2211 User 에게 Item 5를 추천 예측과 2211 User에게 상위 7개 아이템을 추천**

```python
user_id = 2211
item_id = 5
# predict preference of user 2211 to item 5
print(ngcf.predict(user=user_id, item=item_id)[0]*100)
# recommend 7 items for user 2211
arr = ngcf.recommend_user(user=user_id, n_rec=7).get(user_id)
```

```python
42.98396408557892
array([   40,  3105, 70325,  5346,   644, 37161, 13188])

#-- 임베딩 레이어를 32로 증가 + 크로스 엔트로피 사용
85.98052859306335
array([  33, 1567,  644, 2542, 2551, 1782, 2540]
""'
40    Miss Zukas and the Raven's Dance
Name: Title, dtype: object
3105    Transvergence
Name: Title, dtype: object
70325    Sophocles Papas: The Guitar, His Life
Name: Title, dtype: object
5346    A New Song (Mitford Years (Paperback))
Name: Title, dtype: object
644    Out of the Deep I Cry : A Clare Fergusson/Russ...
Name: Title, dtype: object
37161    Tears of the Night Sky (Dragonlance Chaos Wars...
Name: Title, dtype: object
13188    The Ark
Name: Title, dtype: object
"""
```

### LightGCN

라이브러리에서 설명하는 LightGCN evaluate의 인자들의 기본값은 다음과 같다.

```python
evaluate(
    model=Books_lightgcn,
    data=test_data,
    neg_sampling=True,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

- loss: 모델이 예측한 값과 실제 값의 차이를 측정하는 지표
- ROC-AUC: 이진 분류 문제에서 많이 사용되는 평가지표
- Precision: 모델이 추천한 상위 n개의 아이템 중 실제로 선호하는 아이템의 비율을 나타내는 지표
- Recall: 실제 선호하는 아이템 중 모델이 추천한 상위 n개의 아이템의 비율을 나타내는 지표
- NDCG: 추천 시스템에서 상위 n개의 추천 아이템이 실제 선호하는 아이템에 얼마나 가까운지 측정

```python
- 'loss': 0.7724095080219578,
- 'roc_auc': 0.8023822472142028,
- 'precision': 0.003142831265283942,
- 'recall': 0.015095409642973187,
- 'ndcg': 0.01568260847208047}
User 2211에게 아이템 5를 추천한 결과
- 96.57081961631775
- {2211: array([ 819,  206, 1567, 1200,  629,  643, 3120])}
```

</div>
</details>

<details>
<summary> 예제 데이터(NGCF, LightGCN) </summary>
<div markdown="1">   

### NGCF

라이브러리에서 설명하는 evaluate 함수의 인자들의 기본값은 다음과 같다.

```python
evaluate(
    model=movie_ngcf,
    data=movie_test_data,
    neg_sampling=True,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

- metrics
    
    - loss: 모델이 예측한 값과 실제 값의 차이를 측정하는 지표
    - ROC-AUC: 이진 분류 문제에서 많이 사용되는 평가지표
    - Precision: 모델이 추천한 상위 n개의 아이템 중 실제로 선호하는 아이템의 비율을 나타내는 지표
    - Recall: 실제 선호하는 아이템 중 모델이 추천한 상위 n개의 아이템의 비율을 나타내는 지표
    - NDCG: 추천 시스템에서 상위 n개의 추천 아이템이 실제 선호하는 아이템에 얼마나 가까운지 측정
    
    ```python
    - 'loss': 0.5240759073181087,
    - 'roc_auc': 0.8248748577474521,
    -'precision': 0.007502708559046588,
    - 'recall': 0.03739828759936866,
    - 'ndcg': 0.03341032193692315}
    ```
    

**2211 User 에게 Item 5를 추천 예측과 2211 User에게 상위 7개 아이템을 추천**

```python
user_id = 2211
item_id = 5
# predict preference of user 2211 to item 110
print(movie_ngcf.predict(user=user_id, item=item_id)[0]*100)
# recommend 7 items for user 2211
arr = movie_ngcf.recommend_user(user=user_id, n_rec=7).get(user_id)
arr
```

```python
38.170212507247925
array([1234,  913,  969,  919, 1270, 2396, 1230])
```

### LightGCN

라이브러리에서 설명하는 LightGCN evaluate의 인자들의 기본값은 다음과 같다.

```python
evaluate(
    model=Movies_lightgcn,
    data=movie_test_data,
    neg_sampling=True,
    metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
)
```

- loss: 모델이 예측한 값과 실제 값의 차이를 측정하는 지표
- ROC-AUC: 이진 분류 문제에서 많이 사용되는 평가지표
- Precision: 모델이 추천한 상위 n개의 아이템 중 실제로 선호하는 아이템의 비율을 나타내는 지표
- Recall: 실제 선호하는 아이템 중 모델이 추천한 상위 n개의 아이템의 비율을 나타내는 지표
- NDCG: 추천 시스템에서 상위 n개의 추천 아이템이 실제 선호하는 아이템에 얼마나 가까운지 측정

```python
- eval log_loss: 0.7936
- eval roc_auc: 0.7750
- eval precision@10: 0.0079
- eval recall@10: 0.0379
- eval ndcg@10: 0.0386

User 2211에게 아이템 5를 추천한 결과
- 84.41452383995056
- {2211: array([2858,  260, 2028, 1210, 1196, 1270, 2997])}
```

</div>
</details>

**2211 User 에게 Item 5를 추천 예측과 2211 User에게 상위 7개 아이템을 추천**

| 커스텀 데이터 | 2211 ⇒ Item 5 | User 2211 |
| --- | --- | --- |
| NGCF | 42.9%
 | array([  40,  3105, 70325,  5346,   644, 37161, 13188]) |
| LightGCN | 96% | array([ 819,  206, 1567, 1200,  629,  643, 3120])} |

| 예제 데이터 | 2211 ⇒ Item 5 | User 2211 |
| --- | --- | --- |
| NGCF | 31.5%
 | array([ 913, 1252,  912, 1148, 1073,  919,  923]) |
| LightGCN | 84.4% | array([2858,  260, 2028, 1210, 1196, 1270, 2997])} |

<!-- 
<details>
<summary>  </summary>
<div markdown="1">   

</div>
</details> -->