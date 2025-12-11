--------------25.08.07
poly_approx.py : 폐기
npy_to_parquet : 폐기
approx.ipynb : 폐기
soc_slope_synthesis : parquet input npy로 바꿔 해보려했으나 폐기

--------------25.08.08

--------25.08.12 SoC_synthesize patch note------

[panli data]
1) SoC 합성시 휴지기간 오탐색 하는 문제 처리.

2) 방전후에도 Clean data 의 SoC가 0보다 높은값에 수렴하는 데이터가 있음
    > >데이터 합성 시 max(0,new value) 로 했을 시 빨간 동그라미 친 부분 같은 현상 일어나는 경우 있음

    >> discharge 구간 이후의 값으로 대체.

-------------------------------------------------