# 표면 결함 검출 — 적응형 임계값 (v2)

## 개요

[v1 (CPU baseline)](../AnomalyDetection-cpu-baseline) 의 후속. v1은 모든 픽셀에 **같은 고정 임계값**을 썼지만, 실험 결과 어떤 고정 파라미터로도 과검·미검을 동시에 낮추지 못했습니다(과검이 34.8 % 벽에 막힘).

원인은 픽셀 위치마다 "정상일 때의 변동 폭"이 다르다는 것입니다 — 매끈한 영역(σ 작음)과 무늬·인쇄 영역(σ 큼)에 같은 잣대를 댈 수 없습니다. v2는 **픽셀별 정상 표준편차 σ** 를 학습해, 절대 밝기차가 아니라 **"정상 변동 대비 몇 σ 벗어났는가"** 로 판정합니다.

## 파이프라인

v1과 동일하되 **기준 학습과 판정 기준**이 다릅니다.

```
train:  정상 이미지 → 픽셀별 평균(mean)과 표준편차(sigma) 맵
        σ² = E[X²] − (E[X])²   (합·제곱합을 함께 누적)
detect: score = |입력 − mean| / max(sigma, sigma_min)
        결함 ⟺ score > k        (k = σ 배수, 정규분포 3σ 관례)
        → 이후 블러 → 이진화 → 모폴로지 → 연결요소는 v1과 동일
```

σ가 큰 무늬 영역은 score가 작아져 자동으로 관대해지고, σ가 작은 매끈한 영역은 엄격해집니다.

## 결과

MVTec AD capsule · 정상 219장 학습 → test 132장(정상 23 / 결함 109) · Intel i5-6600, Release x64.

**실험 D — k 스윕** ([`experiments/exp_D_k_sweep.csv`](experiments/exp_D_k_sweep.csv))

**같은 미검 수준에서 v2의 과검이 낮습니다:**

| 미검 ≈ | v1 고정 과검 | v2 적응형 과검 | 개선 |
|---|---:|---:|---:|
| ~2 % | 91.3 % (t=30) | 82.6 % (k=2.5) | −8.7 %p |
| ~5 % | 87.0 % (t=35) | **65.2 % (k=3.5)** | **−21.8 %p** |
| ~13–21 % | 69.6 % (t=40) | 56.5 % (k=4.0) | 과검·미검 모두 개선 |

**남은 한계:** k를 올려도 과검이 다시 34.8 %에 고정됩니다 — 이는 무늬가 아닌 **정렬 오차·조명 차이**가 원인이라는 신호이며, 다음 개선은 정합(registration)입니다.

**비용:** 픽셀별 σ 나눗셈(float)이 추가돼 처리시간 **~7.5 ms → ~23 ms (~45 FPS)**. 정확도를 위해 속도를 지불한 트레이드오프.

## 빌드 · 실행

Visual Studio 2022, vcpkg OpenCV(x64), **Release · x64**.
```
AnomalyDetection-adaptive.exe <capsule_root> [k] [sigma_min] [min_area] [blur_ksize] [morph_ksize]
  기본값: k=3.0  sigma_min=3.0  min_area=50  blur_ksize=5  morph_ksize=5
```

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — CC BY-NC-SA 4.0. **저장소에 미포함** ([v1 README](../AnomalyDetection-cpu-baseline/README.md#데이터셋) 참고).
