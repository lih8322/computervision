# 표면 결함 검출 — CPU 베이스라인 (Anomaly Detection)

정상 제품 이미지의 **평균(골든 이미지)** 을 기준으로, 입력이 기준에서 얼마나 벗어났는지를 계산해 표면 결함의 위치를 찾는 프로그램입니다. 산업 이상탐지 데이터셋 **MVTec AD** 로 검증했습니다.

> 딥러닝이 아닌 **고전 영상처리**로 구현했습니다. 검사 파이프라인의 각 단계가 "무슨 문제를 푸는지"를 이해하고 직접 제어하는 것이 목표였고, 이후 GPU(CUDA) 포팅의 성능 비교 기준선(baseline)으로 삼습니다.

---

## 접근 방식 — 왜 이상탐지인가

불량은 종류가 무한하고 드물어 모두 수집하기 어렵지만, 정상 제품은 확보하기 쉽습니다. 그래서 **정상만 학습해 "정상의 기준"을 세우고, 거기서 벗어난 것을 결함으로 판정**합니다. 결함 종류를 미리 알 필요가 없습니다. (MVTec AD의 `train`에 정상 이미지만 있는 이유이기도 합니다.)

---

## 파이프라인

```
train:  정상 이미지 N장 → 그레이스케일 → 평균 → 기준 이미지(reference)

detect: 입력 이미지
          │  그레이스케일 · 크기 정렬
          ├─ ① absdiff        입력과 기준의 절대 차분  → "차이 지도"
          ├─ ② GaussianBlur   한두 픽셀 노이즈 완화
          ├─ ③ threshold      회색조 → 흑백 (흰색 = 결함 후보)
          ├─ ④ morphology     opening(노이즈 제거) → closing(구멍 메움)
          └─ ⑤ CCL            연결요소로 결함 덩어리 → 면적 필터 → 결함 박스
```

| 단계 | 목적 | 핵심 선택 |
|---|---|---|
| 그레이스케일 | 결함은 밝기 이탈 → 색 불필요 | 채널 3→1 (연산·메모리 1/3) |
| absdiff | 기준과 다른 곳 검출 | subtract 아닌 **절대값** — 밝은/어두운 결함 양방향 |
| blur | 노이즈 억제 | 작고 고립된 건 죽고, 뭉친 결함은 삼 |
| threshold | 결함/정상 확정 | 임계값 = 검출 민감도 (미검↔과검 조절) |
| morphology | 형태 정리 | **open→close 순서** — 노이즈를 먼저 치워야 뭉치지 않음 |
| CCL | 덩어리 세기·위치 | 8-연결(비스듬한 스크래치 온전히), 면적 필터로 최종 노이즈 제거 |

---

## 빌드

- **Visual Studio 2022** (MSVC v143), **x64 / Release**
- **OpenCV 4.12** (vcpkg `x64-windows`) — `core`, `imgproc`, `imgcodecs`
- C++17 (`std::filesystem`), 소스 UTF-8

`ComputerVision.sln` 을 열고 구성을 **Release · x64** 로 두고 빌드합니다. (vcpkg가 OpenCV DLL을 출력 폴더에 자동 배포)

## 실행

```
AnomalyDetection-cpu-baseline.exe <capsule_root>
```
`<capsule_root>` 아래에 `train/good/`(정상)과 `test/<종류>/`(검사 대상)이 있어야 합니다.

---

## 결과

MVTec AD **capsule**, 정상 219장으로 학습 → test 132장 검출.
측정 환경: Intel Core i5-6600 @ 3.30GHz, Windows, MSVC Release x64 (단일 스레드).

| 지표 | 값 |
|---|---|
| 처리시간 mean | **7.82 ms/frame** |
| p50 / p95 / p99 | 6.58 / 12.3 / 17.7 ms |
| 처리량 | **약 128 FPS** |

### 검출 품질에 대한 정직한 메모

현재 기본 파라미터(threshold=30)에서는 **정상 이미지까지 결함으로 잡히는 과검**이 많습니다(132장 중 128장 결함 판정 — 정상 23장 상당수 포함). 원인은 두 가지입니다.

1. **고정 임계값의 한계** — 캡슐 표면의 무늬·인쇄처럼 정상끼리도 변동이 큰 영역은, 매끈한 영역과 같은 임계값을 쓰면 정상 변동이 결함으로 잡힙니다.
2. **골든 이미지 방식의 전제** — 입력과 기준의 정렬·조명이 일정해야 하며, 어긋나면 정상 부위도 차분에서 튑니다.

즉 이 베이스라인은 **속도 기준선과 파이프라인 검증**이 목적이며, 검출 품질 튜닝은 아래 로드맵에서 다룹니다.

---

## 로드맵

- [ ] **파라미터 튜닝** — threshold·min_area를 결함 종류별로 조정, 정상/결함 분리도(ROC) 측정
- [ ] **적응형 임계값** — 픽셀별 표준편차(σ) 기반 `k·σ` 임계값 (매끈한 곳 엄격, 무늬 있는 곳 관대)
- [ ] **자동 임계값** — Otsu 등으로 고정값 제거
- [ ] **GPU 가속 (v2)** — absdiff·blur·threshold·morphology를 CUDA 커널로 포팅, 본 CPU 베이스라인과 처리시간 비교
- [ ] **정렬 보정** — 입력–기준 정합(registration)으로 과검 감소

---

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 라이선스 **CC BY-NC-SA 4.0**(연구·비상업용). **데이터는 이 저장소에 포함하지 않습니다.** 위 링크에서 받아 아래 구조로 두세요.

```
capsule/
├── train/good/                    정상 (기준 학습용)
└── test/{good,crack,poke,...}/    검사 대상
```

> P. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
