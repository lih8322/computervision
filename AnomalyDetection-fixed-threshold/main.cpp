// ─────────────────────────────────────────────────────────────────────────
//  표면 결함 검출 — CPU 베이스라인 (Anomaly Detection)
//
//  전략(골든 이미지 / 이상탐지): 정상 이미지들의 평균으로 "기준 이미지"를
//  만들고, 입력이 기준에서 얼마나 벗어났는지(차분)로 결함을 찾는다.
//  정상만 학습하므로 결함 종류를 미리 알 필요가 없다.
//
//  파이프라인: train(정상 평균) → detect(absdiff → blur → threshold →
//              morphology → connected components)
// ─────────────────────────────────────────────────────────────────────────
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>   // OpenCV 로그 레벨 제어

// Windows 콘솔 출력을 UTF-8로 맞추기 위함.
// NOMINMAX: windows.h의 min/max 매크로가 std::max와 충돌하는 것 방지.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════
//  결함 검출기 — ★ 여기 두 함수(train, detect)를 직접 채운다 ★
//  개념은 STUDY_NOTES.md 참고. OpenCV 함수만 우리가 공부한 순서대로 호출하면 됨.
// ═══════════════════════════════════════════════════════════════════════════
class DefectDetector {
public:
    // ── 파라미터 (튜닝 대상: threshold·min_area 등을 바꿔가며 결과 관찰) ──
    struct Params {
        int    blur_ksize  = 5;     // 가우시안 커널 (홀수)
        double threshold   = 30.0;  // 차분 이진화 임계값 (검출 민감도)
        int    morph_ksize = 5;     // 모폴로지 커널
        int    min_area    = 50;    // 이보다 작은 덩어리는 노이즈로 버림
    };
    Params params;

    // ─────────────────────────────────────────────────────────────────────
    //  train: 정상 이미지 여러 장 → 평균 → 기준 이미지(reference_)
    // ─────────────────────────────────────────────────────────────────────
    void train(const std::vector<cv::Mat>& normals) {
        if (normals.empty()) return;

        // 1) 첫 이미지 크기를 기준으로 삼는다.
        const cv::Size sz = normals.front().size();

        // 2) 누적 버퍼: 32F(=CV_32FC1). 여러 장을 더하면 합이 255를 넘어
        //    8U로는 오버플로 → 넉넉한 float에 쌓는다.
        cv::Mat acc = cv::Mat::zeros(sz, CV_32FC1);

        int n = 0;
        for (const auto& img : normals) {
            // GRAY 변환: 결함은 밝기 이탈이라 색 불필요 → 1채널로 줄여 단순·경량화
            cv::Mat gray;
            to_gray(img, gray);
            // 크기가 다르면 맞춘다: acc += f 는 같은 좌표끼리 더하므로 격자가 같아야 함
            if (gray.size() != sz) cv::resize(gray, gray, sz);
            // 8U → 32F 로 맞춰 누적 (acc가 32F라 타입이 같아야 더해짐)
            cv::Mat f;
            gray.convertTo(f, CV_32FC1);
            acc += f;
            ++n;
        }

        // 3) 합 ÷ 장수 = 각 픽셀의 평균 밝기 (정상일 때의 기준값)
        acc /= std::max(1, n);

        // 4) 계산은 float로 했지만, 결과는 표준 이미지 형식(8U)으로 되돌린다.
        //    이후 detect()의 absdiff(입력8U, reference_)와 타입이 맞아야 하므로.
        acc.convertTo(reference_, CV_8UC1);
        ref_size_ = sz;
    }

    // ─────────────────────────────────────────────────────────────────────
    //  detect: 입력 한 장 → 결함 박스들
    // ─────────────────────────────────────────────────────────────────────
    std::vector<cv::Rect> detect(const cv::Mat& input) {
        std::vector<cv::Rect> defects;
        if (reference_.empty() || input.empty()) return defects;

        // ② 입력 전처리: 기준과 같은 채널(GRAY)·크기로 맞춰야 같은 좌표끼리 연산 가능
        cv::Mat gray;
        to_gray(input, gray);
        if (gray.size() != ref_size_) cv::resize(gray, gray, ref_size_);

        // ③ 차분: diff = |입력 − 기준|. 정상은 0에 가깝고 결함만 밝게 튄다.
        //    subtract가 아닌 absdiff — 결함은 기준보다 밝을 수도 어두울 수도 있어
        //    부호 없이 "얼마나 다른가"만 봐야 양방향 다 잡힌다.
        cv::Mat diff;
        cv::absdiff(gray, reference_, diff);

        // ④ 블러: 한두 픽셀짜리 노이즈(센서·정렬오차)를 뭉개 제거.
        //    작고 고립된 건 사라지고, 넓게 뭉친 진짜 결함은 살아남는다.
        cv::Mat blurred;
        cv::GaussianBlur(diff, blurred,
                         cv::Size(params.blur_ksize, params.blur_ksize), 0);

        // ⑤ 이진화: 회색조 차이지도 → 흑백 확정(흰색=결함). 임계값이 검출 민감도.
        cv::Mat bin;
        cv::threshold(blurred, bin, params.threshold, 255, cv::THRESH_BINARY);

        // ⑥ 모폴로지: opening(노이즈 청소) → closing(결함 구멍 메움).
        //    순서 중요 — 노이즈를 먼저 지워야 close에서 뭉쳐 가짜 결함이 안 된다.
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(params.morph_ksize, params.morph_ksize));
        cv::morphologyEx(bin, bin, cv::MORPH_OPEN,  kernel);
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);

        // ⑦ 연결요소: 붙어있는 흰 픽셀을 덩어리로 묶어 면적·위치 추출(8-연결).
        cv::Mat labels, stats, centroids;
        int ncomp = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);
        for (int i = 1; i < ncomp; ++i) {          // 0 = 배경이라 1부터
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < params.min_area) continue;   // 너무 작으면 노이즈로 버림
            defects.emplace_back(
                stats.at<int>(i, cv::CC_STAT_LEFT),
                stats.at<int>(i, cv::CC_STAT_TOP),
                stats.at<int>(i, cv::CC_STAT_WIDTH),
                stats.at<int>(i, cv::CC_STAT_HEIGHT));
        }
        return defects;
    }

    bool trained() const { return !reference_.empty(); }

private:
    // 컬러면 GRAY로, 이미 1채널이면 그대로. (결함검출은 밝기만 보면 됨)
    static void to_gray(const cv::Mat& in, cv::Mat& out) {
        if (in.channels() == 1) out = in.clone();
        else cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    }

    cv::Mat  reference_;   // 정상 평균 (기준 이미지)
    cv::Size ref_size_;
};

// ═══════════════════════════════════════════════════════════════════════════
//  배관(plumbing) — 이미 채워둠. 파일 로딩·측정·main 구조.
//  나중에 이해되면 읽어두면 좋지만, 지금 당장 방어 대상은 위 알고리즘.
// ═══════════════════════════════════════════════════════════════════════════

// 폴더 안의 이미지 파일들을 읽어 vector<Mat>로 반환
static std::vector<cv::Mat> load_images(const fs::path& dir) {
    std::vector<cv::Mat> imgs;
    if (!fs::exists(dir)) return imgs;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file()) files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
        cv::Mat img = cv::imread(f.string(), cv::IMREAD_COLOR);
        if (!img.empty()) imgs.push_back(img);
    }
    return imgs;
}

int main(int argc, char** argv) {
    // 콘솔을 UTF-8로 (소스가 UTF-8이라 안 맞추면 한글이 CP949로 깨짐)
    SetConsoleOutputCP(CP_UTF8);
    // OpenCV의 INFO 로그(TBB 플러그인 등) 숨기기
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <capsule_root> [threshold] [min_area] [blur_ksize] [morph_ksize]\n"
                  << "  기본값: threshold=30  min_area=50  blur_ksize=5  morph_ksize=5\n"
                  << "  (blur_ksize, morph_ksize는 홀수여야 함)\n"
                  << "  예: " << argv[0] << " ...\\capsule 60 200\n";
        return 1;
    }
    const fs::path root = argv[1];

    // ── 파라미터: 인자로 받아 재빌드 없이 실험 (없으면 기본값) ──
    DefectDetector det;
    auto arg_i = [&](int idx, int def) {
        return (argc > idx) ? std::atoi(argv[idx]) : def;
    };
    auto arg_d = [&](int idx, double def) {
        return (argc > idx) ? std::atof(argv[idx]) : def;
    };
    det.params.threshold   = arg_d(2, det.params.threshold);
    det.params.min_area    = arg_i(3, det.params.min_area);
    det.params.blur_ksize  = arg_i(4, det.params.blur_ksize);
    det.params.morph_ksize = arg_i(5, det.params.morph_ksize);

    // 커널 크기는 홀수여야 함(짝수면 +1 보정)
    if (det.params.blur_ksize  % 2 == 0) ++det.params.blur_ksize;
    if (det.params.morph_ksize % 2 == 0) ++det.params.morph_ksize;

    std::cout << "파라미터: threshold=" << det.params.threshold
              << "  min_area="   << det.params.min_area
              << "  blur_ksize=" << det.params.blur_ksize
              << "  morph_ksize=" << det.params.morph_ksize << "\n";

    // 1) 정상 이미지로 기준 학습
    auto normals = load_images(root / "train" / "good");
    std::cout << "정상 이미지 " << normals.size() << "장 로드\n";
    if (normals.empty()) {
        std::cerr << "ERROR: " << (root / "train" / "good") << " 에 이미지가 없습니다.\n";
        return 1;
    }

    det.train(normals);
    if (!det.trained()) {
        std::cerr << "아직 train()이 비어 있습니다. TODO ①을 구현하세요.\n";
        return 1;
    }

    // 2) test 하위 폴더들을 돌며 검출 + 시간 측정
    const fs::path test_dir = root / "test";
    if (!fs::exists(test_dir)) {
        std::cerr << "ERROR: " << test_dir << " 없음\n";
        return 1;
    }

    std::vector<double> times_ms;
    int total = 0;

    // 폴더별 집계: {폴더명, 전체 장수, 결함으로 판정된 장수}
    struct FolderStat { std::string name; int count = 0; int flagged = 0; };
    std::vector<FolderStat> folder_stats;

    // 과검/미검 누적 (good=정상, 나머지=결함)
    int good_total = 0, good_flagged = 0;      // 정상인데 결함 판정 → 과검
    int defect_total = 0, defect_flagged = 0;  // 결함인데 결함 판정 → 정답

    // 헤더: 한글은 콘솔에서 2칸 폭이라 setw가 어긋남 → 공백으로 직접 정렬
    std::cout << "\n폴더               장수    결함판정\n";

    for (const auto& sub : fs::directory_iterator(test_dir)) {
        if (!sub.is_directory()) continue;
        const std::string name = sub.path().filename().string();
        const bool is_good = (name == "good");

        auto imgs = load_images(sub.path());
        FolderStat fs_stat{ name, 0, 0 };

        for (const auto& img : imgs) {
            auto t0 = std::chrono::steady_clock::now();
            auto defects = det.detect(img);
            auto t1 = std::chrono::steady_clock::now();
            times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

            const bool flagged = !defects.empty();  // 결함으로 판정?
            ++total;
            ++fs_stat.count;
            if (flagged) ++fs_stat.flagged;

            if (is_good) { ++good_total;   if (flagged) ++good_flagged; }
            else         { ++defect_total; if (flagged) ++defect_flagged; }
        }
        folder_stats.push_back(fs_stat);

        std::cout << std::left << std::setw(16) << name
                  << std::right << std::setw(8) << fs_stat.count
                  << std::setw(12) << fs_stat.flagged
                  << (is_good ? "  <- 정상(0이 이상적)" : "") << "\n";
    }

    // 과검/미검 요약
    const int miss = defect_total - defect_flagged;  // 결함인데 못 잡음 = 미검
    std::cout << "\n----- 검출 품질 -----\n";
    std::cout << "과검 (정상 오검) : " << good_flagged << "/" << good_total;
    if (good_total)   std::cout << "  (" << (100.0 * good_flagged / good_total) << "%)";
    std::cout << "\n";
    std::cout << "미검 (결함 놓침) : " << miss << "/" << defect_total;
    if (defect_total) std::cout << "  (" << (100.0 * miss / defect_total) << "%)";
    std::cout << "\n";

    // 3) 측정 결과 (mean / p50 / p95 / p99 / FPS)
    if (!times_ms.empty()) {
        std::sort(times_ms.begin(), times_ms.end());
        auto pct = [&](double p) {
            size_t i = static_cast<size_t>(p / 100.0 * (times_ms.size() - 1));
            return times_ms[i];
        };
        double sum = 0; for (double t : times_ms) sum += t;
        double mean = sum / times_ms.size();

        std::cout << "\n----- 처리시간 (ms/frame, 총 " << total << "장) -----\n";
        std::cout << "mean : " << mean    << "\n";
        std::cout << "p50  : " << pct(50)  << "\n";
        std::cout << "p95  : " << pct(95)  << "\n";
        std::cout << "p99  : " << pct(99)  << "\n";
        std::cout << "FPS  : " << (1000.0 / mean) << "\n";
    }
    return 0;
}
