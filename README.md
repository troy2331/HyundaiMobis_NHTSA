# HyundaiMobis_NHTSA
# 2022. 03. 30 ~ 2022. 06. 15. 
현대모비스 X NHTSA (미국 도로 교통국) 고객 불만 데이터 토픽 이상징후 모델링 Python 코드 입니다

매일 발생하는 500~1000자의 고객 불만 텍스트 데이터에 대하여, 이상 시기를 제안하여 읽어야할 텍스트를 한정시켜주는 모델입니다. 

데이터 출처 : NHTSA 미국 도로 교통국
누구나 다운받을 수 있습니다. 
https://www-odi.nhtsa.dot.gov/downloads/

## Python 코드 구성 
Data 원본 저장 폴더 
ckpt : Wegiht 저장 폴더 
models : Model 저장 폴더
save : Custom_data 저장 폴더
main.py : Train, Test 기능 
make_data.py : Train_set 제작
make_test.py : test_set 제작 및 LDA 시각화 기능 

## UI 예시 
C#을 이용하여 UI로 연동하면 다음과 같은 인터페이스에서 이상 징후를 탐지하고, 선택 시점에 대하여 LDA모델을 확인할 수 있습니다. 
<img src="https://github.com/troy2331/HyundaiMobis_NHTSA/issues/2#issue-1272219693">
<img src="복사해온 URL">
