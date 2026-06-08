"""
파일 관리 유틸리티 모듈

파일 및 디렉토리 관리 기능을 제공합니다.
"""

import os
import shutil
from typing import Optional
from pathlib import Path


class FileManager:
    """
    파일 및 디렉토리 관리를 위한 유틸리티 클래스
    """

    @staticmethod
    def ensure_directory(path: str) -> None:
        """
        디렉토리가 존재하지 않으면 생성

        Args:
            path: 디렉토리 경로
        """
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def manage_old_files(directory: str, max_files: int = 7) -> None:
        """
        오래된 파일/디렉토리 삭제 (최근 N개만 유지)

        Args:
            directory: 대상 디렉토리 경로
            max_files: 유지할 최대 파일 개수 (기본값: 7)
        """
        if not os.path.exists(directory):
            print(f"디렉토리가 존재하지 않습니다: {directory}")
            return

        file_list = os.listdir(directory)

        if len(file_list) <= max_files:
            print(f"파일 개수가 {max_files}개 이하입니다. 삭제할 파일이 없습니다.")
            return

        # 파일들을 수정 시간 기준으로 정렬
        file_list.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)))

        # 최근 max_files개를 제외한 나머지 선택
        files_to_remove = file_list[:-max_files]

        for file_name in files_to_remove:
            file_path = os.path.join(directory, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"파일 삭제 완료: {file_name}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"디렉토리 삭제 완료: {file_name}")
            except Exception as e:
                print(f"삭제 중 오류 발생 ({file_path}): {e}")

    @staticmethod
    def create_filename_directory(base_path: str, filename: str) -> str:
        """
        파일명 기반 디렉토리 생성

        Args:
            base_path: 기본 경로
            filename: 파일명 (디렉토리명으로 사용)

        Returns:
            생성된 디렉토리 경로
        """
        dir_path = os.path.join(base_path, filename)
        FileManager.ensure_directory(dir_path)
        return dir_path

    @staticmethod
    def get_file_list(directory: str) -> list:
        """
        디렉토리 내 파일 목록 조회

        Args:
            directory: 디렉토리 경로

        Returns:
            파일/디렉토리 이름 리스트
        """
        if not os.path.exists(directory):
            return []
        return os.listdir(directory)

    @staticmethod
    def file_exists(directory: str, filename: str) -> bool:
        """
        파일 또는 디렉토리 존재 여부 확인

        Args:
            directory: 디렉토리 경로
            filename: 파일/디렉토리 이름

        Returns:
            존재 여부
        """
        file_list = FileManager.get_file_list(directory)
        return filename in file_list
