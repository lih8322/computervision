# Computer Vision — 표면 결함 검출 (Anomaly Detection)

OpenCV(C++)로 산업 표면 결함 검출을 직접 구현한 저장소입니다. 정상 이미지 기반 이상탐지 파이프라인을 밑바닥부터 만들고, 파라미터 실험으로 한계를 규명한 뒤 적응형 임계값으로 개선했습니다. 모든 결과는 **MVTec AD** 로 실측했습니다.

## 프로젝트 구성

| 프로젝트 | 내용 |
|---|---|
| [**AnomalyDetection-cpu-baseline**](AnomalyDetection-cpu-baseline) | 정상 평균(골든 이미지) 차분 기반 CPU 파이프라인 + 파라미터 실험 |
| [**AnomalyDetection-adaptive**](AnomalyDetection-adaptive) | 픽셀별 표준편차(σ) 기반 적응형 임계값 — baseline의 한계 극복 |

파이프라인: `정상 평균 → 차분(absdiff) → 블러 → 이진화 → 모폴로지 → 연결요소`
지표: **과검**(정상을 결함이라 함) / **미검**(결함을 놓침), 폴더별 집계. 원본 수치는 각 프로젝트 `experiments/` CSV.

## 결과 요약

MVTec AD capsule · 정상 219장 학습 → test 132장(정상 23 / 결함 109) · Intel i5-6600, MSVC Release x64.

| | 과검 | 미검 | 처리시간 |
|---|---:|---:|---:|
| v1 고정 임계값 (균형점) | 34.8 % | 37.6 % | ~7.5 ms |
| v2 적응형 (동일 미검 기준) | −22 %p 개선 | 6.4 % | ~23 ms |

- 고정 파라미터(threshold·min_area·blur)로는 과검·미검을 동시에 낮추지 못함 → 픽셀별 적응형 임계값으로 같은 미검 수준에서 과검 최대 ~22 %p 감소.
- 남은 과검(34.8 %)은 무늬가 아닌 정렬·조명 문제로 판단 → 다음은 정합(registration).

## 기술 스택

- **C++17**, **OpenCV 4.12** (vcpkg `x64-windows`)
- **Visual Studio 2022** (MSVC v143), Release · x64
- 통계(평균·분산)·공간 필터(가우시안·모폴로지)·연결요소 분석

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — CC BY-NC-SA 4.0. **저장소에 포함하지 않습니다.**
> P. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
