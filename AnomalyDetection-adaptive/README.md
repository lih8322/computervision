# 표면 결함 검출 — 적응형 임계값 (v2)

## 개요

[v1 (CPU baseline)](../AnomalyDetection-fixed-threshold) 의 후속입니다. v1은 모든 픽셀에 같은 고정 임계값을 썼는데, 실험에서 어떤 고정 파라미터로도 과검·미검이 함께 낮아지지 않고 과검이 34.8 %에서 멈췄습니다.

픽셀 위치마다 정상일 때의 변동 폭이 다른 점(매끈한 영역은 σ 작음, 무늬·인쇄 영역은 σ 큼)이 원인으로 보여, v2에서는 픽셀별 정상 표준편차 σ를 학습해 절대 밝기차 대신 "정상 변동 대비 몇 σ 벗어났는가"로 판정하도록 바꿔 보았습니다.

## 파이프라인

v1과 같되 기준 학습과 판정 기준이 다릅니다.

```
train:  정상 이미지 → 픽셀별 평균(mean)과 표준편차(sigma) 맵
        σ² = E[X²] − (E[X])²   (합·제곱합을 함께 누적)
detect: score = |입력 − mean| / max(sigma, sigma_min)
        결함 ⟺ score > k        (k = σ 배수, 정규분포 3σ 관례)
        → 이후 블러 → 이진화 → 모폴로지 → 연결요소는 v1과 동일
```

σ가 큰 무늬 영역은 score가 작아지고, σ가 작은 매끈한 영역은 커지는 구조입니다.

## 결과

MVTec AD capsule · 정상 219장 학습 → test 132장(정상 23 / 결함 109) · Intel i5-6600, Release x64.

**실험 D — k 스윕** ([`experiments/exp_D_k_sweep.csv`](experiments/exp_D_k_sweep.csv))

같은 미검 수준끼리 비교하면 v2의 과검이 더 낮게 나왔습니다:

| 미검 ≈ | v1 고정 과검 | v2 적응형 과검 | 차이 |
|---|---:|---:|---:|
| ~2 % | 91.3 % (t=30) | 82.6 % (k=2.5) | −8.7 %p |
| ~5 % | 87.0 % (t=35) | **65.2 % (k=3.5)** | **−21.8 %p** |
| ~13–21 % | 69.6 % (t=40) | 56.5 % (k=4.0) | 과검·미검 모두 낮음 |

**남은 한계:** k를 올려도 과검이 다시 34.8 %에서 멈춥니다. 이 부분은 무늬보다 정렬 오차·조명 차이 쪽이 원인으로 보이며, 다음에 정합(registration)을 시도해 볼 지점입니다.

**비용:** 픽셀별 σ 나눗셈(float)이 더해져 처리시간이 **~7.5 ms → ~23 ms (~45 FPS)** 로 늘었습니다.

## 빌드 · 실행

Visual Studio 2022, vcpkg OpenCV(x64), **Release · x64**.
```
AnomalyDetection-adaptive.exe <capsule_root> [k] [sigma_min] [min_area] [blur_ksize] [morph_ksize]
  기본값: k=3.0  sigma_min=3.0  min_area=50  blur_ksize=5  morph_ksize=5
```

## 데이터셋

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — CC BY-NC-SA 4.0. **저장소에 미포함** ([v1 README](../AnomalyDetection-fixed-threshold/README.md#데이터셋) 참고).
