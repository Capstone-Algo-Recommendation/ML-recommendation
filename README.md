# algorithm_problem_recomm
 "사용자가 풀 수 있는 문제 중 가장 어려운 문제와 비슷한 문제들을 추천"

## 1. data
* **solved.ac**에서 제공하는 데이터 모으기
  * **problemMeta.csv**: problemID, title, tags, acceptedUser, averageTries, level
  * **problemTagMeta.csv**: key, bojTagID, problemCount, name_kor, name_eng
  * **solvedProblem.csv**: handle, count, problemIds
* **EDA**
  * **문제**
    * 22,768개의 문제, 0-30의 난이도
    * 문제 난이도가 0인 것은 평가되지 않은 문제
    * 평가되지 않은 문제 수: 8144개
    * 아주 쉽거나, 아주 어려운 문제는 그렇게 많지 않음
  * **문제 태그**
    * 187가지의 태그
    * 태그가 없는 문제 수: 8988개
    * 문제에서 나타난 태그 조합은 3763개
    * '수학-구현-다이나믹프로그래밍-자료구조-그래프이론' 순으로 태그 빈도가 높음
  * **사용자**
    * 59,898 명의 사용자
    * 문제를 15개 이하로 맞춘 사람이 전체의 18% 정도 차지
    * **maxlevel**: 푼 문제 중 가장 어려운 문제의 level
      * 10-16인 사용자가 가장 많음
 * **Dataset 구성**
    * **maxlevel**를 기준으로 **5개의 cluster** 분할
    * **cluster별 Dataset 구성**
      * for cluster1: cluster1+cluster2
      * for cluster2: cluster2+cluster3
      * ...
      * for cluster5: cluster5
    * **Dataset Split**
      * **train data**: Dataset의 0.8
        * 푼 문제가 15개 이하인 경우는 train data에서 제외시킴
      * **validation/test data**: Dataset의 0.1
  
## 2. Model
* 잠재요인 협업필터링 기반 모델
* 평가지표: Recall@30, Hit Rate@30
* **MF**
  * **학습 정보**
    * optimizer: SGD
    * learning rate: 1e-2
    * Batch 학습 진행 (batch size: 1024)
    * Hit Rate@30이 0.3을 넘어가면 학습 종료
    * epoch: 20회 내외
  * **성능**
    * Recall@30: 0.041
    * Hit Rate@30: 0.397
* **NCF**
  * **학습 정보**
    * RandomSearch를 이용한 하이퍼파라미터 튜닝 진행
      * 잠재요인 수, learning rate, Dropout1, Dropout2 
    * optimizer: Adam
    * Batch 학습 진행 (batch size: 1024)
    * epoch: 5회 내외
  * **성능**
    * Recall@30: 0.137
    * Hit Rate@30: 0.749
