# 12주차 12주차 (NGCF metrics 분석, Recommender FrameWork)

### 라이브러리에서 사용하는 NGCF  Metrics 분석

1. Pointwise: Pointwise 방법론은 각 사용자-아이템 쌍에 대해 개별적으로 예측 값을 생성하고 평가하는 방법입니다. 이 방법론에서는 각 쌍의 예측 값과 실제 평가 값 사이의 차이를 최소화하거나 평가 지표를 최적화하는 것이 목표입니다. 대표적인 예로는 평점 예측 문제가 있습니다. 사용자-아이템 쌍에 대해 평점을 예측하고, 실제 평점과의 차이를 최소화하여 모델을 학습합니다.
2. Listwise: Listwise 방법론은 추천 목록 전체에 대해 평가하고 최적화하는 방법입니다. 이 방법론에서는 추천 목록의 순서를 고려하여 평가 지표를 최적화하거나 추천 목록 자체를 최적화하는 것이 목표입니다. Listwise 방법론은 사용자의 선호도 순서를 모델에 반영하여 개별적인 아이템 예측보다 전체 목록의 품질을 개선하는 데 초점을 둡니다. 예를 들어, 검색 결과의 순위매기기나 추천 시스템에서의 Top-N 추천 문제에 사용될 수 있습니다.

Pointwise와 Listwise 방법론은 모델의 학습과 평가 방식에서 차이가 있습니다. Pointwise는 개별 예측 값에 대해 최적화를 수행하고, Listwise는 전체 추천 목록에 대한 최적화를 수행합니다. 선택은 문제의 성격과 데이터의 특성에 따라 달라질 수 있습니다

```python
eval_result = evaluate(
    model=movie_ngcf,
    data=movie_test_data,
    neg_sampling=True,
    metrics=["loss", "precision", "recall"],
    k = 10
)
```

```python
POINTWISE_METRICS = {
    "loss",
    "log_loss",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "roc_gauc",
}
LISTWISE_METRICS = {"precision", "recall", "map", "ndcg", "coverage"}
RANKING_METRICS = POINTWISE_METRICS | LISTWISE_METRICS
```

- **precision_at_k 함수**
    
    ```python
    def precision_at_k(y_true, y_reco, k):
        common_items = set(y_reco).intersection(y_true)
        #-- 추천된 아이템 중 실제로 사용자가 선호하는(구매하거나 클릭한) 아이템의 갯수 / k
        return len(common_items) / k
    ```
    
    **TF/(TF + FT)  → 전체 아이템 중 실제 아이템 비율을 다시 K로 나눔**
    

**#-- 실제값들과 예측값, 샘플링된 유저들, k를 인자로 listwise_scores 계산**

- **listwise_scores(fn, y_trues, y_recos, users, k) 함수**
    
    ```python
    scores = list()
    for u in users:
    	y_true = y_true_lists[u]
    	y_reco = y_reco_lists[u]
    	scores.append(fn(y_true, y_reco, k))
    return np.mean(scores)
    ```
    
    precision_at_k를 통해 얻은 값들을 scores 리스트에 모두 저장 후 평균값 계산

### LightGCN 프레임워크 만들기
- [Colab에서 구현](https://colab.research.google.com/drive/17B2jHE-yasJaz_hkzl_6eeUi_zH8xd_c?usp=sharing")
