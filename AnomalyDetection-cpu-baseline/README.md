# 표면 결함 검출 — CPU 베이스라인

## 개요

정상 제품 이미지의 **평균(골든 이미지)** 을 기준으로, 입력이 기준에서 얼마나 벗어났는지를 계산해 표면 결함을 찾습니다. 불량은 종류가 무한하고 드물지만 정상은 확보하기 쉬우므로, **정상만 학습해 벗어난 것을 결함으로 판정**하는 이상탐지 방식입니다. 딥러닝이 아닌 고전 영상처리로 파이프라인을 직접 구현했습니다.

## 파이프라인

```
train:  정상 이미지 N장 → 그레이스케일 → 평균 → 기준 이미지(reference)
detect: 입력 → ① absdiff(기준과 차분) → ② 블러(노이즈 완화)
             → ③ 이진화(threshold) → ④ 모폴로지(open→close)
             → ⑤ 연결요소(면적 필터) → 결함 박스
```

| 단계 | 핵심 |
|---|---|
| 그레이스케일 | 결함은 밝기 이탈 → 색 불필요, 연산 1/3 |
| absdiff | subtract 아닌 절대값 — 밝은/어두운 결함 양방향 |
| threshold | 임계값 = 검출 민감도 (과검↔미검 조절) |
| 모폴로지 | open(노이즈 제거)→close(구멍 메움) 순서 |
| 연결요소 | 8-연결, 면적 필터로 노이즈 최종 제거 |

## 결과

MVTec AD capsule · 정상 219장 학습 → test 132장(정상 23 / 결함 109) · Intel i5-6600, Release x64.
처리시간 **~7.5 ms/frame (~130 FPS)**. 지표: 과검(정상 오검) / 미검(결함 놓침).

**파라미터 실험** — 한 번에 하나씩 변경. 전체 수치는 [`experiments/RESULTS.md`](experiments/RESULTS.md) (원본 CSV 포함):

| 실험 | 관찰 |
|---|---|
| A. threshold 25→70 | 과검↔미검 시소, 균형점(t=50)도 둘 다 ~35 % |
| B. min_area 50→400 | 과검 34.8 %에서 변화 없음, 미검만 악화 |
| C. blur 5→11 | 과검 34.8 %에서 변화 없음, 미검만 악화 |

**결론:** 단일 고정 파라미터로는 과검·미검을 동시에 낮출 수 없습니다. 과검의 원인이 작은 노이즈가 아니라 정상끼리도 변동이 큰 무늬 영역이기 때문입니다. → 픽셀별 적응형 임계값을 [**v2 (adaptive)**](../AnomalyDetection-adaptive) 에서 구현해 같은 미검 수준에서 과검을 최대 ~22 %p 낮췄습니다.

## 빌드 · 실행

Visual Studio 2022, vcpkg OpenCV(x64), **Release · x64**.
```
AnomalyDetection-cpu-baseline.exe <capsule_root> [threshold] [min_area] [blur_ksize] [morph_ksize]
  기본값: threshold=30  min_area=50  blur_ksize=5  morph_ksize=5
```

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — CC BY-NC-SA 4.0. **저장소에 미포함.** `capsule/train/good/`(정상)과 `capsule/test/<종류>/`(검사 대상) 구조로 두고 실행합니다.
> P. Bergmann et al., "MVTec AD," CVPR 2019.
