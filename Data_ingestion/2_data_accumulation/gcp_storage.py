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
