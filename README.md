# Computer Vision — 표면 결함 검출 (Anomaly Detection)

OpenCV(C++) 기반 산업 표면 결함 검출 저장소입니다. 정상 이미지 기반 이상탐지 파이프라인을 구현하고, 파라미터 실험으로 한계를 규명한 뒤 적응형 임계값·정렬로 단계적으로 개선해 보았습니다. 모든 결과는 **MVTec AD** 로 실측했습니다.

## 프로젝트 구성

| 프로젝트 | 내용 |
|---|---|
| [**AnomalyDetection-cpu-baseline**](AnomalyDetection-cpu-baseline) (v1) | 정상 평균(골든 이미지) 차분 기반 CPU 파이프라인 + 파라미터 실험 |
| [**AnomalyDetection-adaptive**](AnomalyDetection-adaptive) (v2) | 픽셀별 표준편차(σ) 기반 적응형 임계값(Z-score) |
| [**AnomalyDetection-registration**](AnomalyDetection-registration) (v3) | 차분 전 ECC 정렬 추가 — 정렬 오차의 영향 검증 |

파이프라인: `정상 평균 → 차분(absdiff) → 블러 → 이진화 → 모폴로지 → 연결요소`
지표: **과검**(정상을 결함이라 함) / **미검**(결함을 놓침), 폴더별 집계. 원본 수치는 각 프로젝트 `experiments/` CSV.

## 결과 요약

MVTec AD capsule · 정상 219장 학습 → test 132장(정상 23 / 결함 109) · Intel i5-6600, MSVC Release x64.
같은 미검 수준끼리 비교한 과검(정상 오검):

| | 과검 | 처리시간 | 비고 |
|---|---:|---:|---|
| v1 고정 임계값 | 기준 | ~7.5 ms | threshold·min_area·blur 스윕 — 어느 것도 과검·미검 동시 개선 못 함 |
| v2 적응형(Z-score) | 최대 −22 %p | ~23 ms | 픽셀별 σ 반영, 무늬 영역 과검 일부 해소 |
| v3 정렬(ECC) | 엄격 구간 −8.7 %p | ~52 ms | 정렬 오차는 일부 원인일 뿐, 과검 34.8 % 벽 남음 |

세 단계를 거치며 소프트웨어 처리만으로는 과검이 34.8 %에서 더 내려가지 않았습니다. 남은 부분은 알고리즘보다 조명·촬영 조건 쪽에 원인이 있는 것으로 보이며, 이 지점에서 광학 통제가 필요합니다. 이 값은 **단일 골든 이미지 차분 방식의 구조적 한계**를 보여주며, 실무에서는 이 지점부터 광학 조건 개선 또는 학습 기반(딥러닝) 방법이 병행됩니다. 검사 방법마다 촬영 통제 의존도가 다르므로(차분 > 특징 기반 > 학습 기반), 환경에 맞춰 방법을 선택하게 됩니다.

## 기술 스택

- **C++17**, **OpenCV 4.12** (vcpkg `x64-windows`)
- **Visual Studio 2022** (MSVC v143), Release · x64
- 통계(평균·분산)·공간 필터(가우시안·모폴로지)·연결요소 분석

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — CC BY-NC-SA 4.0. **저장소에 포함하지 않습니다.**
> P. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
