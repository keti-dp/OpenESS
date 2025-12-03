"""
Katib 하이퍼파라미터 튜닝 파이프라인
Max Voltage 모델 (RACK_MAX_CELL_VOLTAGE) 학습
"""
from kfp import compiler, dsl
from kfp.dsl import component, pipeline, Output, Dataset


@component(
    base_image='python:3.12',
    packages_to_install=['kubeflow-katib', 'kubernetes', 'google-cloud-storage']
)
def run_katib_experiment(
    namespace: str,
    experiment_name_prefix: str,
    target_column: str,
    timeout: int,
    max_trial_count: int,
    parallel_trial_count: int,
    parameters_config: str,
    training_image: str,
    early_stopping_enabled: bool,
    early_stopping_algorithm: str,
    early_stopping_min_trials: int,
    early_stopping_start_step: int,
    best_params_output: Output[Dataset]
):
    """
    Katib 하이퍼파라미터 튜닝 실험 실행

    데이터 소스: PVC 'ess-dataset' (/mnt/ess-dataset/{site_id}/data에 마운트)
    결과 저장: PVC (/mnt/ess-dataset/{site_id}/models/yyyymm/)
    """
    import json
    import time
    from kubeflow.katib import KatibClient
    from kubeflow.katib import V1beta1Experiment, V1beta1ExperimentSpec
    from kubeflow.katib import V1beta1AlgorithmSpec, V1beta1ObjectiveSpec
    from kubeflow.katib import V1beta1ParameterSpec, V1beta1FeasibleSpace
    from kubeflow.katib import V1beta1TrialTemplate, V1beta1TrialParameterSpec
    from kubeflow.katib import V1beta1EarlyStoppingSpec
    from kubernetes import client as k8s_client

    # 컴포넌트 내부에서 실험 이름 생성 (실행 시간 포함)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_name = f"{experiment_name_prefix}-max-voltage-{timestamp}"

    # GCP Credential Secret 이름 (K8s에 미리 생성된 Secret 사용)
    secret_name = "gcp-credentials"
    print(f"Katib 실험 시작: {experiment_name}")
    print(f"Target: {target_column}")
    print(f"Timestamp: {timestamp}")
    print(f"GCP Credential Secret: {secret_name}")

    # 파라미터 설정 로드
    params_dict = json.loads(parameters_config)

    # Trial Parameters
    trial_parameters = []
    for model_name, model_params in params_dict.items():
        for param in model_params:
            param_name = f"{model_name}_{param['name']}"
            trial_parameters.append(V1beta1TrialParameterSpec(
                name=param_name,
                description=f"{model_name.upper()} {param['name']}",
                reference=param_name
            ))

    # Objective
    objective = V1beta1ObjectiveSpec(
        type="minimize",
        # goal 제거 - 모든 trial 실행하며 최적값 탐색
        objective_metric_name="rmse",
        additional_metric_names=["mae", "r2"]
    )

    # Algorithm
    algorithm = V1beta1AlgorithmSpec(
        algorithm_name="bayesianoptimization",
        algorithm_settings=[{"name": "random_state", "value": "42"}]
    )

    # Early Stopping (sites.yaml에서 설정)
    early_stopping = None
    if early_stopping_enabled:
        early_stopping = V1beta1EarlyStoppingSpec(
            algorithm_name=early_stopping_algorithm,
            algorithm_settings=[
                {"name": "min_trials_required", "value": str(early_stopping_min_trials)},
                {"name": "start_step", "value": str(early_stopping_start_step)}
            ]
        )
        print(f"Early Stopping 활성화: {early_stopping_algorithm}")
        print(f"  Min trials: {early_stopping_min_trials}, Start step: {early_stopping_start_step}")
    else:
        print("Early Stopping 비활성화")

    # Hyper Parameters
    parameters = []
    for model_name, model_params in params_dict.items():
        for param in model_params:
            param_name = f"{model_name}_{param['name']}"
            feasible_space = V1beta1FeasibleSpace(
                min=str(param['min']),
                max=str(param['max'])
            )
            if 'step' in param:
                feasible_space.step = str(param['step'])

            parameters.append(V1beta1ParameterSpec(
                name=param_name,
                parameter_type=param['type'],
                feasible_space=feasible_space
            ))

    # Trial Template (Docker 이미지 사용)
    args_list = ["--target", target_column]
    for model_name, model_params in params_dict.items():
        for param in model_params:
            param_name = f"{model_name}_{param['name']}"
            arg_name = f"--{param['name'].replace('_', '-')}"
            args_list.extend([arg_name, f"${{trialParameters.{param_name}}}"])

    trial_spec = k8s_client.V1Job(
        api_version="batch/v1",
        kind="Job",
        spec=k8s_client.V1JobSpec(
            template=k8s_client.V1PodTemplateSpec(
                metadata=k8s_client.V1ObjectMeta(
                    annotations={"sidecar.istio.io/inject": "false"}
                ),
                spec=k8s_client.V1PodSpec(
                    containers=[k8s_client.V1Container(
                        name="training-container",
                        image=training_image,
                        image_pull_policy="Always", # 안정적인 image라면 해당 옵션은 "IfNotPresent"로 변경 가능
                        args=args_list,
                        env=[
                            k8s_client.V1EnvVar(
                                name="DATA_PATH",
                                value=f"/mnt/ess-dataset/{experiment_name_prefix}/data"  # PersistentVolume 경로
                            ),
                            # GCS Credentials는 결과 저장용으로만 사용 (데이터 로드는 PV 사용)
                            k8s_client.V1EnvVar(
                                name="GOOGLE_APPLICATION_CREDENTIALS",
                                value="/var/secrets/google/key.json"
                            ),
                            # GCS 모델 저장 설정 (환경변수 사용)
                            k8s_client.V1EnvVar(
                                name="GCS_BUCKET",
                                value="your-gcs-bucket"  # 환경에 맞게 설정
                            ),
                            k8s_client.V1EnvVar(
                                name="GCS_MODEL_PATH",
                                value=f"vt-model/{experiment_name_prefix}/models"
                            ),
                            # Trial 이름 (Downward API로 pod 이름 전달)
                            k8s_client.V1EnvVar(
                                name="TRIAL_NAME",
                                value_from=k8s_client.V1EnvVarSource(
                                    field_ref=k8s_client.V1ObjectFieldSelector(
                                        field_path="metadata.name"
                                    )
                                )
                            )
                        ],
                        volume_mounts=[
                            # PersistentVolume 마운트 (데이터 읽기용)
                            k8s_client.V1VolumeMount(
                                name="ess-dataset",
                                mount_path="/mnt/ess-dataset",
                                read_only=True
                            ),
                            # GCP Credentials 마운트 (결과 저장용)
                            k8s_client.V1VolumeMount(
                                name="gcp-credentials",
                                mount_path="/var/secrets/google",
                                read_only=True
                            )
                        ],
                        resources=k8s_client.V1ResourceRequirements(
                            limits={"memory": "16Gi", "cpu": "2"},
                            requests={"memory": "8Gi", "cpu": "1"}
                        )
                    )],
                    volumes=[
                        # PersistentVolumeClaim (데이터 저장소)
                        k8s_client.V1Volume(
                            name="ess-dataset",
                            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                                claim_name="ess-dataset"
                            )
                        ),
                        # GCP Credentials Secret
                        k8s_client.V1Volume(
                            name="gcp-credentials",
                            secret=k8s_client.V1SecretVolumeSource(
                                secret_name=secret_name
                            )
                        )
                    ],
                    restart_policy="Never",
                    # Only use high-spec nodes (node1~node6)
                    affinity=k8s_client.V1Affinity(
                        node_affinity=k8s_client.V1NodeAffinity(
                            required_during_scheduling_ignored_during_execution=k8s_client.V1NodeSelector(
                                node_selector_terms=[
                                    k8s_client.V1NodeSelectorTerm(
                                        match_expressions=[
                                            k8s_client.V1NodeSelectorRequirement(
                                                key="kubernetes.io/hostname",
                                                operator="In",
                                                values=["node1", "node2", "node4", "node5", "node6", "node7", "node10"]
                                            )
                                        ]
                                    )
                                ]
                            )
                        )
                    )
                )
            )
        )
    )

    trial_template = V1beta1TrialTemplate(
        primary_container_name="training-container",
        trial_parameters=trial_parameters,
        trial_spec=trial_spec,
        retain=False  # 성공한 Trial Pod는 자동 삭제, 실패한 것은 Kubernetes가 더 오래 유지
    )

    # Experiment 생성
    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=k8s_client.V1ObjectMeta(
            name=experiment_name,
            namespace=namespace
        ),
        spec=V1beta1ExperimentSpec(
            algorithm=algorithm,
            objective=objective,
            early_stopping=early_stopping,
            parameters=parameters,
            max_trial_count=max_trial_count,
            parallel_trial_count=parallel_trial_count,
            max_failed_trial_count=min(5, max_trial_count),  # max_trial_count보다 클 수 없음
            trial_template=trial_template
        )
    )

    # Katib 실험 제출
    # 참고: GCP Credential은 K8s Secret 'gcp-credentials'를 사용
    # Secret 생성 방법:
    # kubectl create secret generic gcp-credentials \
    #   --from-file=key.json=/path/to/credential.json \
    #   -n space-openess
    katib_client = KatibClient()

    try:
        katib_client.delete_experiment(name=experiment_name, namespace=namespace)
        print(f"기존 실험 삭제: {experiment_name}")
        time.sleep(5)
    except:
        pass

    katib_client.create_experiment(experiment, namespace=namespace)
    print(f"[OK] Katib experiment created: {experiment_name}")
    print(f"  Max trials: {max_trial_count}, Parallel: {parallel_trial_count}")

    # 완료 대기 (커스텀 polling 로직)
    print(f"완료 대기 중... (최대 {timeout}초 = {timeout//3600}시간)")

    start_time = time.time()
    poll_interval = 30  # 30초마다 체크
    last_status_print = 0

    while True:
        elapsed = time.time() - start_time

        # 타임아웃 체크
        if elapsed > timeout:
            raise TimeoutError(f"Timeout ({timeout}s) waiting for experiment to complete")

        # 실험 상태 조회 (객체를 dictionary로 변환)
        exp_obj = katib_client.get_experiment(name=experiment_name, namespace=namespace)
        exp = exp_obj.to_dict() if hasattr(exp_obj, 'to_dict') else exp_obj
        status = exp.get('status') or {}  # None 처리
        conditions = status.get('conditions') or []  # None 처리

        # 1분마다 상태 출력
        if elapsed - last_status_print >= 60:
            trials = status.get('trials', 0)
            running = status.get('trials_running', 0)
            succeeded = status.get('trials_succeeded', 0)
            failed = status.get('trials_failed', 0)
            print(f"[{int(elapsed)}s] Trials: {trials}, Running: {running}, Succeeded: {succeeded}, Failed: {failed}")
            last_status_print = elapsed

        # Succeeded 또는 Failed 상태 체크
        experiment_finished = False
        for condition in conditions:
            cond_type = condition.get('type', '')
            cond_status = str(condition.get('status', '')).lower()

            if cond_type == 'Succeeded' and cond_status == 'true':
                print(f"\n[OK] Experiment succeeded! (Elapsed: {int(elapsed)}s)")
                experiment_finished = True
                break
            elif cond_type == 'Failed' and cond_status == 'true':
                msg = condition.get('message', 'Unknown error')
                reason = condition.get('reason', 'Unknown reason')
                print(f"\n[ERROR] Experiment failed! (Elapsed: {int(elapsed)}s)")
                print(f"  Reason: {reason}")
                print(f"  Message: {msg}")
                experiment_finished = True
                break

        if experiment_finished:
            break

        # 아직 완료되지 않았으면 대기 후 재시도
        time.sleep(poll_interval)

    # 최종 상태 재조회 (객체를 dictionary로 변환)
    exp_obj = katib_client.get_experiment(name=experiment_name, namespace=namespace)
    exp = exp_obj.to_dict() if hasattr(exp_obj, 'to_dict') else exp_obj
    status = exp.get('status') or {}  # None 처리

    # 상태 정보 출력
    print(f"\n[DEBUG] Experiment status:")
    print(f"  Trials: {status.get('trials', 0)}")
    print(f"  Trials Running: {status.get('trials_running', status.get('trialsRunning', 0))}")
    print(f"  Trials Succeeded: {status.get('trials_succeeded', status.get('trialsSucceeded', 0))}")
    print(f"  Trials Failed: {status.get('trials_failed', status.get('trialsFailed', 0))}")
    print(f"  Conditions: {status.get('conditions') or []}")

    # 실패 확인
    for condition in (status.get('conditions') or []):
        if condition.get('type') == 'Failed' and condition.get('status') == 'True':
            error_msg = condition.get('message', 'Unknown error')
            reason = condition.get('reason', 'Unknown reason')
            print(f"\n[ERROR] Experiment failed!")
            print(f"  Reason: {reason}")
            print(f"  Message: {error_msg}")
            raise RuntimeError(f"Experiment failed: {reason} - {error_msg}")

    # 최적 파라미터 파싱
    optimal_trial = status.get('current_optimal_trial', status.get('currentOptimalTrial', {}))
    if not optimal_trial:
        raise ValueError("최적 trial을 찾을 수 없음")

    # 베스트 trial 이름 추출
    best_trial_name = (
        optimal_trial.get('best_trial_name') or
        optimal_trial.get('name') or
        optimal_trial.get('trialName') or
        optimal_trial.get('trial_name') or
        ''
    )
    print(f"\n[INFO] Best trial: {best_trial_name}")

    if not best_trial_name:
        print(f"[WARNING] Could not extract trial name from optimal_trial")
        print(f"  Available keys: {optimal_trial.keys()}")

    params = {}
    param_assignments = optimal_trial.get('parameter_assignments', optimal_trial.get('parameterAssignments', []))
    for param in param_assignments:
        name, value = param['name'], param['value']
        # 값이 소수점을 포함하면 float, 그렇지 않으면 int로 변환
        try:
            # 먼저 int로 변환 시도 (소수점이 없는 경우)
            if '.' in str(value):
                params[name] = float(value)
            else:
                params[name] = int(value)
        except ValueError:
            # 변환 실패 시 float로 시도
            params[name] = float(value)

    metrics = {}
    observation = optimal_trial.get('observation', {})
    for metric in observation.get('metrics', []):
        metrics[metric['name']] = float(metric['latest'])

    # 결과 출력
    print(f"\n[OK] Experiment succeeded!")
    print(f"Best hyperparameters: {params}")
    print(f"Metrics: {metrics}")

    # 결과 저장 (PVC에 저장)
    from datetime import datetime
    from pathlib import Path
    import os

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    yearmonth = datetime.now().strftime('%Y%m')  # yyyymm

    # 파일명: yyyymm_xgboost_site_id.json
    site_id = experiment_name_prefix  # site_id from experiment name prefix
    filename = f"{yearmonth}_xgboost_{site_id}.json"

    result = {
        'site_id': site_id,
        'target_column': target_column,
        'parameters': params,
        'metrics': metrics,
        'experiment_name': experiment_name,
        'total_trials': status.get('trials', 0),
        'succeeded_trials': status.get('trials_succeeded', status.get('trialsSucceeded', 0)),
        'failed_trials': status.get('trials_failed', status.get('trialsFailed', 0)),
        'timestamp': timestamp,
        'yearmonth': yearmonth,
        'filename': filename
    }

    # 로컬 artifact에 저장 (Kubeflow Pipeline용)
    with open(best_params_output.path, 'w') as f:
        json.dump(result, f, indent=2)

    # GCS에 저장 ({gcs_bucket}/vt-model/{site_id}/models/yyyymm/)
    from google.cloud import storage
    import os
    import tempfile
    import base64

    gcs_bucket_name = os.getenv('GCS_BUCKET', 'your-gcs-bucket')
    model_base_path = os.getenv('GCS_MODEL_BASE_PATH', 'vt-model')
    gcs_path = f"{model_base_path}/{site_id}/models/{yearmonth}/{filename}"

    print(f"\n[INFO] Saving results to GCS...")
    print(f"  Bucket: {gcs_bucket_name}")
    print(f"  Path: {gcs_path}")

    # GCP Credentials 설정 (Kubernetes Secret에서 읽기)
    credentials_file = None
    try:
        from kubernetes import client, config

        # In-cluster config 로드
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        # gcp-credentials secret 읽기
        secret = v1.read_namespaced_secret(
            name="gcp-credentials",
            namespace=namespace
        )

        # key.json 추출 (base64 디코딩)
        key_json = base64.b64decode(secret.data['key.json']).decode('utf-8')

        # 임시 파일로 저장
        credentials_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        credentials_file.write(key_json)
        credentials_file.close()

        # 환경변수 설정
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file.name
        print(f"  ✓ GCP credentials loaded from secret: gcp-credentials")

    except Exception as cred_error:
        print(f"  [WARNING] Failed to load credentials from secret: {cred_error}")
        print(f"  Attempting to use default credentials")

    try:
        # GCS 클라이언트 초기화
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_path)

        # JSON 데이터 업로드
        blob.upload_from_string(
            json.dumps(result, indent=2),
            content_type='application/json'
        )

        print(f"  ✓ Saved: gs://{gcs_bucket_name}/{gcs_path}")
        print(f"[OK] GCS hyperparameters save completed!")

        # 베스트 trial의 모델을 최종 위치로 복사
        if best_trial_name:
            model_filename = f"{yearmonth}_xgboost_{site_id}_model.pkl"
            trials_prefix = f"{model_base_path}/{site_id}/models/{yearmonth}/trials/"
            final_model_path = f"{model_base_path}/{site_id}/models/{yearmonth}/{model_filename}"

            print(f"\n[INFO] Copying best trial model...")
            print(f"  Best trial: {best_trial_name}")

            try:
                    # trials 폴더에서 베스트 trial로 시작하는 모델 파일 찾기 (Pod suffix 고려)
                blobs = list(bucket.list_blobs(prefix=trials_prefix))
                matching_blobs = [
                    blob for blob in blobs
                    if blob.name.startswith(f"{trials_prefix}{best_trial_name}") and blob.name.endswith("_model.pkl")
                ]

                if matching_blobs:
                    source_blob = matching_blobs[0]  # 첫 번째 매칭 파일 사용
                    print(f"  From: gs://{gcs_bucket_name}/{source_blob.name}")
                    print(f"  To:   gs://{gcs_bucket_name}/{final_model_path}")

                    bucket.copy_blob(source_blob, bucket, final_model_path)
                    print(f"  ✓ Model copied successfully!")
                    print(f"[OK] Best model save completed!")
                    print(f"\n[INFO] All trials models are kept in: gs://{gcs_bucket_name}/{trials_prefix}")
                else:
                    print(f"  [WARNING] No model found for trial: {best_trial_name}")
                    print(f"  Searched in: gs://{gcs_bucket_name}/{trials_prefix}")
                    print(f"  Pattern: {best_trial_name}*_model.pkl")
            except Exception as model_error:
                print(f"  [WARNING] Failed to copy model: {model_error}")
        else:
            print(f"\n[WARNING] Best trial name not found, skipping model copy")

    except Exception as e:
        print(f"[WARNING] Failed to save to GCS: {e}")
        print(f"  Results are still available in Kubeflow artifact output")
    finally:
        # 임시 credentials 파일 정리
        if credentials_file and os.path.exists(credentials_file.name):
            try:
                os.unlink(credentials_file.name)
            except:
                pass


@pipeline(
    name='katib-hyperparameter-tuning',
    description='Battery Max Voltage Model Hyperparameter Tuning'
)
def katib_tuning_pipeline(
    namespace: str = 'your-namespace',
    experiment_name_prefix: str = 'default-site',
    timeout: int = 7200,
    max_trial_count: int = 30,
    parallel_trial_count: int = 3,
    parameters_config: str = '{}',
    training_image: str = 'your-training-image:tag',
    early_stopping_enabled: bool = True,
    early_stopping_algorithm: str = 'medianstop',
    early_stopping_min_trials: int = 3,
    early_stopping_start_step: int = 5
):
    """
    Katib Hyperparameter Tuning Pipeline
    Training RACK_MAX_CELL_VOLTAGE prediction model

    Data source: PVC 'ess-dataset' mounted at /mnt/ess-dataset/{site_id}/data
    Training image and early_stopping settings are managed in config/katib_config.yaml
    Best hyperparameters are saved to GCS: gs://{gcs_bucket}/{model_base_path}/{site_id}/models/yyyymm/
    """
    run_katib_experiment(
        namespace=namespace,
        experiment_name_prefix=experiment_name_prefix,
        target_column='RACK_MAX_CELL_VOLTAGE',
        timeout=timeout,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
        parameters_config=parameters_config,
        training_image=training_image,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_algorithm=early_stopping_algorithm,
        early_stopping_min_trials=early_stopping_min_trials,
        early_stopping_start_step=early_stopping_start_step
    )


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=katib_tuning_pipeline,
        package_path='katib_tuning_pipeline.yaml'
    )
    print("[OK] Pipeline compiled: katib_tuning_pipeline.yaml")
