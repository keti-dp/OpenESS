"""
gcp_storage.py : GCS(Google Cloud Storage) 파일 업로드 모듈

Copyright(C) 2021, 윤태일 / KETI / taeil777@keti.re.kr

최신 테스트 버전 : 1.0.0 ver
최신 안정화 버전 : 1.0.0 ver

GCS에 만들어진 bucket에 파일을 업로드하는 모듈 코드입니다.

프로젝트명, 버킷명, gcp 키를 통해 동작시킬 수 있습니다.

전체적인 코드에 대한 설명은 https://github.com/keti-dp/OpenESS 에서 확인하실 수 있습니다.
"""


import sys

sys.path.append("./")
import google_storage_class


_PROJECT_NAME = "gcp프로젝트이름"
_BUCKET_NAME = "bucket이름"
_CREDENTIAL_JSON_FILE_PATH = "gcp키경로"


def upload_2bucket(gcs_file_path, local_file_path):

    # GCS 객체 생성
    GCP = google_storage_class.google_cloud_storage(
        _PROJECT_NAME, _BUCKET_NAME, _CREDENTIAL_JSON_FILE_PATH
    )

    GCP.upload_to_bucket(gcs_file_path, local_file_path)


if __name__ == "__main__":

    upload_2bucket()
