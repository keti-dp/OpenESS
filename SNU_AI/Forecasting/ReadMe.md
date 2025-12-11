
학습데이터 : ./data/SoC_synthesis 에서 20220502까지
평가데이터 : ./data/SoC_synthesis 에서 (20220502, 20220504)

./data/generated
    20220504 원본 데이터 -> 학습된 웨이트로 output
                        -> 끝쪽에 ReLu처럼 해서 0으로 밀어버림
                        -> 초반에 constant부분처럼 강제로 맞추어 둠.
                        -> generated에 저장.