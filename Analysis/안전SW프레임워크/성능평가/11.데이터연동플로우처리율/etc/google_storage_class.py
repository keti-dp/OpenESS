# -*-coding:utf-8-*-

from google.cloud import storage
from google.oauth2 import service_account
import sys
import time
import os

class google_cloud_storage:
    # google_cloud_storage 객체 생성
    def __init__(self, _PROJECT_NAME, _BUCKET_NAME, _CREDENTIAL_JSON_FILE_PATH=None):
        if _CREDENTIAL_JSON_FILE_PATH != None:
            self.credentials = service_account.Credentials.from_service_account_file(_CREDENTIAL_JSON_FILE_PATH)
        else:
            # 인증관련 환경변수 등록 (https://cloud.google.com/docs/authentication/getting-started#auth-cloud-implicit-python)
            self.credentials = None
        self.client = storage.Client(project = _PROJECT_NAME, credentials = self.credentials)
        self.bucket_name = _BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)
    

    # client의 bucket 재지정
    def select_bucket(self, _BUCKET_NAME):
        self.bucket_name = _BUCKET_NAME
        self.bucket = self.client.bucket(self.bucket_name)
        

    # client의 모든 bucket 리스트 반환
    def get_all_bucket(self):
        buckets = self.client.list_buckets()
        for bucket in buckets:
            print(bucket.name)
        return buckets


    # 현재 지정된 client의 bucket 정보 출력
    def print_bucket_info(self):
        print("ID: {}".format(self.bucket.id))
        print("Name: {}".format(self.bucket.name))
        print("Storage Class: {}".format(self.bucket.storage_class))
        print("Location: {}".format(self.bucket.location))
        print("Location Type: {}".format(self.bucket.location_type))
        print("Cors: {}".format(self.bucket.cors))
        print(
            "Default Event Based Hold: {}".format(self.bucket.default_event_based_hold)
        )
        print("Default KMS Key Name: {}".format(self.bucket.default_kms_key_name))
        print("Metageneration: {}".format(self.bucket.metageneration))
        print(
            "Retention Effective Time: {}".format(
                self.bucket.retention_policy_effective_time
            )
        )
        print("Retention Period: {}".format(self.bucket.retention_period))
        print("Retention Policy Locked: {}".format(self.bucket.retention_policy_locked))
        print("Requester Pays: {}".format(self.bucket.requester_pays))
        print("Self Link: {}".format(self.bucket.self_link))
        print("Time Created: {}".format(self.bucket.time_created))
        print("Versioning Enabled: {}".format(self.bucket.versioning_enabled))
        print("Labels: %s" % self.bucket.labels)


    # 새 버킷 생성
    def create_new_bucket(self, _BUCKET_NAME):
        new_bucket = self.client.create_bucket(_BUCKET_NAME)
        print("Bucket {} created.".format(new_bucket.name))
        return new_bucket


    # 현재 client 버킷의 모든 객체 반환 (_prefix : 버킷내의 경로 필터링)
    def get_all_blob(self, _prefix=None):
        blobs = client.list_blobs(self.bucket, _prefix)
        for blob in blobs:
            print(blob.name)
        return blobs


    # 현재 client 버킷에 특정 파일 업로드
    def upload_to_bucket(self, destination_blob_name, source_file_path):
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        print(
        "File {} uploaded to {}.".format(
            source_file_path, destination_blob_name
            )
        )


    # 현재 client 버킷에 특정 객체(파일) 다운로드
    def download_blob(self, source_blob_name, destination_file_name):
        blob = self.bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(
            "Blob {} downloaded to {}.".format(
                source_blob_name, destination_file_name
            )
        )

