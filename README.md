# Lezhin Data Challenge PyConKr 2017

## Link
http://tech.lezhin.com/events/data-challenge-pyconkr-2017

## Initial features
  * 1 : label 해당 유저가 목록에 진입하고 1시간 이내에 구매했는지 여부 (ORDERED)
  * 2 : 사용 플랫폼 A (PLATFORM_A)
  * 3 : 사용 플랫폼 B (PLATFORM_B)
  * 4 : 사용 플랫폼 C (PLATFORM_C)
  * 5 : 사용 플랫폼 D (PLATFORM_D)
  * 6 : 목록 진입시점 방문 총 세션 수 (범위별로 부여된 순차 ID) (SESSION_CNT)
  * 7 : 작품을 나타내는 해쉬 (PRODUCT_ID)
  * 8-10 : 개인정보 (USER_ID_1 ~ 3)
  * 11-110 : 주요 작품 구매 여부 (BUY_PRODUCT_1 ~ 100)
  * 111 : 작품 태그 정보 (TAG)
  * 112 : 구매할 때 필요한 코인 (COIN_NEEDED)
  * 113 : 완결 여부 (COMPLETED)
  * 114-123 : 스케쥴 정보 (SCHEDULE_1 ~ 10)
  * 124-141 : 장르 정보 (GENRE_1 ~ 18)
  * 142 : 해당 작품의 마지막 에피소드 발행 시점 (범위별로 부여된 순차 ID) (LAST_EPISODE)
  * 143 : 단행본 여부 (PUBLISHED)
  * 144 : 작품 발행 시점 (범위별로 부여된 순차 ID) (START_DATE)
  * 145 : 총 발행 에피소드 수 (범위별로 부여된 순차 ID) (TOTAL_EPISODE_CNT)
  * 146-151 : 작품 태그 정보 (TAG_1 ~ 5)
  * 152-167 : 유저의 성향 정보 (과거에 구매를 했을 때만 기록) (TENDENCY_1 ~ 16)
  * I removed some 'less important' initial features.

## Additional features
  * 1 product feature
    * PRD_ORDERED_SUM
  * 3 user_product features
    * UP_VIEW_CNT
    * UP_ORDERED_SUM
    * UP_ORDERED_RATIO
  * 1 combined feature
    * UP_PRD_ORDERED_RATIO

## Requirements
  * python==3.6.2
  * numpy==1.13.1
  * pandas==0.20.3
  * scikit-learn==0.18.1
  * xgboost==0.6
