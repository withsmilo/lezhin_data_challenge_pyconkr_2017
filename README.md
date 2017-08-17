# Lezhin Data Challenge PyConKr 2017

## Link
http://tech.lezhin.com/events/data-challenge-pyconkr-2017

## Initial features
  * 1 : label 해당 유저가 목록에 진입하고 1시간 이내에 구매했는지 여부
  * 2 : 사용 플랫폼 A
  * 3 : 사용 플랫폼 B
  * 4 : 사용 플랫폼 C
  * 5 : 사용 플랫폼 D
  * 6 : 목록 진입시점 방문 총 세션 수 (범위별로 부여된 순차 ID)
  * 7 : 작품을 나타내는 해쉬
  * 8-10 : 개인정보
  * 11-110 : 주요 작품 구매 여부
  * 111 : 작품 태그 정보
  * 112 : 구매할 때 필요한 코인
  * 113 : 완결 여부
  * 114-123 : 스케쥴 정보
  * 124-141 : 장르 정보
  * 142 : 해당 작품의 마지막 에피소드 발행 시점 (범위별로 부여된 순차 ID)
  * 143 : 단행본 여부
  * 144 : 작품 발행 시점 (범위별로 부여된 순차 ID)
  * 145 : 총 발행 에피소드 수 (범위별로 부여된 순차 ID)
  * 146-151 : 작품 태그 정보
  * 152-167 : 유저의 성향 정보 (과거에 구매를 했을 때만 기록)

## Additional features
  * https://github.com/withsmilo/lezhin_data_challenge_pyconkr_2017/blob/master/src/features.py
  * 34 user features
  * 6 product features
  * 3 user_product features

## Requirements
  * python==3.6.2
  * numpy==1.13.1
  * pandas==0.20.3
  * scikit-learn==0.18.1
  * xgboost==0.6
