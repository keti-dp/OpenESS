"""
KFP 2.x용 Kubeflow 클라이언트
Python 3.12 호환
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import kfp
from kfp import Client


class KubeflowClient:
    """Kubeflow Pipelines와 상호작용하는 클라이언트"""

    def __init__(self, host=None, username=None, password=None, namespace=None):
        """
        Kubeflow 클라이언트 초기화

        Args:
            host: Kubeflow 호스트 URL
            username: 인증용 사용자 이름
            password: 인증용 비밀번호
            namespace: Kubernetes namespace
        """
        # Load from .env if parameters not provided
        if not all([host, username, password, namespace]):
            env_path = Path(__file__).parent.parent / 'config' / '.env'
            if env_path.exists():
                load_dotenv(env_path)

        self.host = host or os.getenv('HOST')
        self.username = username or os.getenv('USERNAME')
        self.password = password or os.getenv('PASSWORD')
        self.namespace = namespace or os.getenv('NAMESPACE')

        if not all([self.host, self.username, self.password, self.namespace]):
            raise ValueError("Missing required credentials. Set HOST, USERNAME, PASSWORD, NAMESPACE")

        # Create client
        self.client = self._create_client()

    def _create_client(self):
        """인증된 Kubeflow 클라이언트 생성"""
        try:
            # Get session cookie first
            session_cookie = self._get_session_cookie()
            if not session_cookie:
                raise ValueError("Failed to obtain session cookie. Check credentials.")

            print(f"✓ Session cookie obtained: {session_cookie[:50]}...")

            # For KFP 2.x with Dex authentication
            # Add /pipeline endpoint like in KFP 1.8.x
            pipeline_host = f"{self.host.rstrip('/')}/pipeline"
            print(f"✓ Connecting to: {pipeline_host}")

            client = Client(
                host=pipeline_host,
                namespace=self.namespace,
                cookies=session_cookie
            )

            return client
        except Exception as e:
            print(f"Error creating Kubeflow client: {e}")
            raise

    def _get_session_cookie(self):
        """Get authentication session cookie (KFP 1.8.x style)"""
        import requests
        import time
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        try:
            print(f"🔐 Attempting to authenticate with Kubeflow...")
            print(f"   Host: {self.host}")
            print(f"   Username: {self.username[:3]}***")  # 보안을 위해 일부만 표시

            # Create session with retry strategy
            session = requests.Session()

            # Retry 설정: 최대 3회, 백오프 전략 사용
            retry_strategy = Retry(
                total=3,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Step 1: Get auth URL with longer timeout
            print(f"   Step 1: Getting auth URL (timeout: 30s)...")
            response = session.get(self.host, verify=False, timeout=30)
            print(f"   Auth URL: {response.url}")
            print(f"   Response status: {response.status_code}")

            # Step 2: Post login credentials
            print(f"   Step 2: Posting credentials (timeout: 30s)...")
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            user_data = {
                "login": self.username,
                "password": self.password
            }

            login_response = session.post(
                response.url,
                headers=headers,
                data=user_data,
                verify=False,
                timeout=30
            )
            print(f"   Login response: {login_response.status_code}")

            # Check if redirect happened (successful login usually redirects)
            if login_response.history:
                print(f"   Redirects: {[r.status_code for r in login_response.history]}")

            # Get authservice_session cookie
            session_cookies = session.cookies.get_dict()
            print(f"   Cookies received: {list(session_cookies.keys())}")

            if "authservice_session" in session_cookies:
                session_cookie = session_cookies["authservice_session"]
                cookie_str = f"authservice_session={session_cookie}"
                print(f"   ✓ Authentication successful!")
                return cookie_str

            # 504 에러 또는 쿠키 없음 - 재시도
            print(f"   ⚠️ No authservice_session cookie received")
            print(f"   Login response status: {login_response.status_code}")
            print(f"   Login response headers: {dict(login_response.headers)}")

            # 응답 내용 일부 출력 (디버깅용)
            if login_response.text:
                print(f"   Response preview: {login_response.text[:200]}...")

            return None

        except requests.exceptions.Timeout as e:
            print(f"❌ Timeout error: {e}")
            print(f"   Kubeflow 서버가 응답하지 않습니다. 서버 상태를 확인하세요.")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error: {e}")
            print(f"   Kubeflow 서버에 연결할 수 없습니다. 네트워크를 확인하세요.")
            return None
        except Exception as e:
            print(f"❌ Could not get session cookie: {e}")
            import traceback
            traceback.print_exc()
            return None

    def upload_pipeline(self, pipeline_path, pipeline_name=None):
        """
        Kubeflow에 파이프라인 업로드

        Args:
            pipeline_path: 컴파일된 파이프라인 YAML 경로
            pipeline_name: 파이프라인 이름 (default: filename)

        Returns:
            파이프라인 ID
        """
        if not pipeline_name:
            pipeline_name = Path(pipeline_path).stem

        try:
            pipeline = self.client.upload_pipeline(
                pipeline_package_path=pipeline_path,
                pipeline_name=pipeline_name
            )
            print(f"파이프라인 업로드 완료: {pipeline_name} (ID: {pipeline.pipeline_id})")
            return pipeline.pipeline_id
        except Exception as e:
            print(f"파이프라인 업로드 오류: {e}")
            raise

    def create_run(self, pipeline_id=None, pipeline_path=None, experiment_name='Default',
                   run_name=None, params=None):
        """
        파이프라인 생성 및 실행

        Args:
            pipeline_id: 기존 파이프라인 ID (optional)
            pipeline_path: Path to pipeline YAML (optional)
            experiment_name: Experiment name
            run_name: 실행 이름
            params: 파이프라인 파라미터 dict

        Returns:
            실행 객체
        """
        if not run_name:
            from datetime import datetime
            run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        params = params or {}

        try:
            if pipeline_path:
                run = self.client.create_run_from_pipeline_package(
                    pipeline_file=pipeline_path,
                    arguments=params,
                    run_name=run_name,
                    experiment_name=experiment_name,
                    namespace=self.namespace
                )
            elif pipeline_id:
                run = self.client.run_pipeline(
                    experiment_id=self._get_or_create_experiment(experiment_name),
                    job_name=run_name,
                    pipeline_id=pipeline_id,
                    params=params
                )
            else:
                raise ValueError("Either pipeline_id or pipeline_path must be provided")

            print(f"실행 생성: {run_name} (ID: {run.run_id})")
            return run
        except Exception as e:
            print(f"실행 생성 오류: {e}")
            raise

    def _get_or_create_experiment(self, experiment_name):
        """실험 가져오기 또는 생성"""
        try:
            experiment = self.client.get_experiment(experiment_name=experiment_name)
            return experiment.experiment_id
        except:
            experiment = self.client.create_experiment(
                name=experiment_name,
                namespace=self.namespace
            )
            return experiment.experiment_id

    def get_run_status(self, run_id):
        """
        실행 상태 가져오기

        Args:
            run_id: Run ID

        Returns:
            실행 상태 dict
        """
        try:
            run = self.client.get_run(run_id)
            return {
                'run_id': run.run_id,
                'status': run.state,
                'created_at': run.created_at,
                'finished_at': run.finished_at,
                'error': run.error if hasattr(run, 'error') else None
            }
        except Exception as e:
            print(f"실행 상태 가져오기 오류: {e}")
            raise

    def wait_for_run_completion(self, run_id, timeout=3600):
        """
        실행 완료 대기

        Args:
            run_id: Run ID
            timeout: 타임아웃(초)

        Returns:
            최종 실행 상태
        """
        try:
            run = self.client.wait_for_run_completion(run_id, timeout=timeout)
            status = self.get_run_status(run_id)
            print(f"실행 완료, 상태: {status['status']}")
            return status
        except Exception as e:
            print(f"실행 대기 오류: {e}")
            raise

    def list_pipelines(self, page_size=10):
        """
        파이프라인 목록

        Args:
            page_size: 반환할 파이프라인 수

        Returns:
            파이프라인 목록
        """
        try:
            pipelines = self.client.list_pipelines(page_size=page_size)
            return pipelines.pipelines
        except Exception as e:
            print(f"파이프라인 목록 오류: {e}")
            return []

    def list_runs(self, experiment_name=None, page_size=10):
        """
        실행 목록

        Args:
            experiment_name: 실험 이름으로 필터링
            page_size: 반환할 실행 수

        Returns:
            실행 목록
        """
        try:
            if experiment_name:
                experiment = self.client.get_experiment(experiment_name=experiment_name)
                runs = self.client.list_runs(
                    experiment_id=experiment.experiment_id,
                    page_size=page_size
                )
            else:
                runs = self.client.list_runs(page_size=page_size)

            return runs.runs
        except Exception as e:
            print(f"실행 목록 오류: {e}")
            return []


def main():
    """Kubeflow 클라이언트 테스트"""
    print("Kubeflow 클라이언트 테스트 중...")

    # Create client
    client = KubeflowClient()

    # 파이프라인 목록
    print("\n파이프라인 목록:")
    pipelines = client.list_pipelines()
    for pipeline in pipelines:
        print(f"  - {pipeline.name} (ID: {pipeline.pipeline_id})")

    # 실행 목록
    print("\n최근 실행 목록:")
    runs = client.list_runs()
    for run in runs:
        print(f"  - {run.name} (Status: {run.state})")


if __name__ == '__main__':
    main()
