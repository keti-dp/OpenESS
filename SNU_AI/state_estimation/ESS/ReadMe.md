작업중 이슈사항 및 공유가 필요한 내용은 이곳에 자유롭게 적어주시기 바랍니다.

*자유양식*
ex.
<제목>
--내용--

<CosineAnnealingWarmup>
cosine-annealing-warmup @ git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup@12d03c07553aedd3d9e9155e2b3e31ce8c64081a
이 패키지를 받아서 scheduler로 사용중. 깃헙에가서 보고 다운 받으시길

<data에 관하여...>
경로 : /data/ess/data/incell/state_estimation_data/(sionyu or panli)/(preprocessed or ocv_labeled)
parquet 파일로 저장
    - preprocessed : interpolated data
    - ocv_labeled : /home/ess/State_Estimation/preprocess.py 의 get_dates 에 있는 날을 제외한 데이터 삭제 및 남은 날들의 ocv labeled data

sionyu, panli
columns = ['TIMESTAMP', 'BANK_DC_VOLT', 'BANK_DC_CURRENT', 'BANK_SOC', 'MAX_CELL_TEMPERATURE_OF_BANK', 'VOLT_gap', 'OCV_est', 'BATTERY_STATUS_FOR_CHARGE']
-> 'BATTERY_STATUS_FOR_CHARGE' : 1(rest), 2(charge), 3(discharge)

normalize X
'TIMESTAMP' : 제대로 사용하려면 우리나라 시간으로 맞춰야함

<columns>
sionyu, panli의 column이름을 통일하려합니다.
앞으로 황금6호 등의 데이터가 들어오므로 통일은 필수입니다.
이름 통일 및 변경은 Dataset에서 진행할테니 앞으로는 모든 코드에서 통일된 컬럼명을 사용하시기 바랍니다.
통일된 컬럼명은 preprocess.py 에서 확인하실수 있습니다.