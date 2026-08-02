# 표면 결함 검출 — 고정 임계값 (v1)

## 개요

정상 제품 이미지의 평균(골든 이미지)을 기준으로, 입력이 기준에서 얼마나 벗어났는지를 계산해 표면 결함을 찾는 이상탐지 방식입니다. 정상만 학습해 기준을 만들고, 벗어난 영역을 결함 후보로 봅니다. 고전 영상처리 기반이며, OpenCV를 익히며 만든 학습 프로젝트입니다.

## 파이프라인

```
train:  정상 이미지 N장 → 그레이스케일 → 평균 → 기준 이미지(reference)
detect: 입력 → ① absdiff(기준과 차분) → ② 블러(노이즈 완화)
             → ③ 이진화(threshold) → ④ 모폴로지(open→close)
             → ⑤ 연결요소(면적 필터) → 결함 박스
```

| 단계 | 메모 |
|---|---|
| 그레이스케일 | 밝기만 사용, 3채널→1채널 |
| absdiff | 절대값 차분 (밝은/어두운 결함 모두) |
| threshold | 임계값이 검출 민감도를 좌우 |
| 모폴로지 | open → close 순서로 적용 |
| 연결요소 | 8-연결 + 면적 필터 |

## 결과

MVTec AD capsule · 정상 219장 학습 → test 132장(정상 23 / 결함 109) · Intel i5-6600, Release x64.
처리시간 **~7.5 ms/frame (~130 FPS)**. 지표: 과검(정상 오검) / 미검(결함 놓침).

**파라미터 실험** — 한 번에 하나씩 변경. 전체 수치는 [`experiments/RESULTS.md`](experiments/RESULTS.md) (원본 CSV 포함):

| 실험 | 관찰 |
|---|---|
| A. threshold 25→70 | 과검↔미검이 반대로 움직임, 균형점(t=50)도 둘 다 ~35 % |
| B. min_area 50→400 | 과검 34.8 %에서 변화 없음, 미검만 악화 |
| C. blur 5→11 | 과검 34.8 %에서 변화 없음, 미검만 악화 |

세 실험 모두 과검이 34.8 %에서 더 내려가지 않았습니다. 단일 고정 파라미터를 조절하는 것만으로는 과검·미검을 함께 낮추기 어려웠고, 정상끼리도 변동이 큰 무늬 영역이 원인으로 보였습니다. 이를 픽셀별로 다루는 적응형 임계값을 [v2 (adaptive)](../AnomalyDetection-adaptive) 에서 시도했습니다.

## 빌드 · 실행

Visual Studio 2022, vcpkg OpenCV(x64), **Release · x64**.
```
AnomalyDetection-fixed-threshold.exe <capsule_root> [threshold] [min_area] [blur_ksize] [morph_ksize]
  기본값: threshold=30  min_area=50  blur_ksize=5  morph_ksize=5
```

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — CC BY-NC-SA 4.0. **저장소에 미포함.** `capsule/train/good/`(정상)과 `capsule/test/<종류>/`(검사 대상) 구조로 두고 실행합니다.
> P. Bergmann et al., "MVTec AD," CVPR 2019.
