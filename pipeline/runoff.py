"""SCS-CN(NRCS-CN) 직접유출 계산.

CN 값: 국토교통부「홍수량 산정 표준지침」(2019) 표 3.1 국내 유출곡선지수 기준(AMC-II)을
환경부 토지피복 중분류 코드에 맞춰 옮겼다. 2025년 세분류 토지피복지도의 일부 코드는
지침 작성 당시 분류와 이름이 달라 아래처럼 가장 가까운 항목을 썼다.
  420 인공초지 → 지침의 골프장(420)·기타초지(430): 49/69/79/84
  610 자연나지 → 지침의 채광지역(610, 개활지 불량): 68/79/86/89
  620 인공나지 → 지침의 기타나지(620, 개발 중인 지역): 77/86/91/94
  230 시설재배지 → 지침의 하우스재배지(230): 76/85/89/91

유출고 Q(mm) = (P - Ia)^2 / (P - Ia + S),  S = 25400/CN - 254,  Ia = 0.2 S  (P ≤ Ia 이면 0)
"""
import numpy as np

# 중분류 코드: (A, B, C, D)  — AMC-II
CN_II = {
    "110": (77, 85, 90, 92),   # 주거지역
    "120": (81, 88, 91, 93),   # 공업지역
    "130": (89, 92, 94, 95),   # 상업지역
    "140": (49, 69, 79, 84),   # 문화·체육·휴양지역 (지침: 위락시설지역)
    "150": (83, 89, 92, 93),   # 교통지역
    "160": (61, 75, 83, 87),   # 공공시설지역
    "210": (79, 79, 79, 79),   # 논 (담수 상태 → 토양군 무관)
    "220": (63, 74, 82, 85),   # 밭
    "230": (76, 85, 89, 91),   # 시설재배지
    "240": (70, 79, 84, 88),   # 과수원
    "250": (68, 79, 86, 89),   # 기타재배지
    "310": (55, 72, 82, 85),   # 활엽수림 (산림 별도 기준)
    "320": (55, 72, 82, 85),   # 침엽수림
    "330": (55, 72, 82, 85),   # 혼효림
    "410": (30, 58, 71, 78),   # 자연초지
    "420": (49, 69, 79, 84),   # 인공초지
    "510": (100, 100, 100, 100),  # 내륙습지 (수면 기준)
    "610": (68, 79, 86, 89),   # 자연나지
    "620": (77, 86, 91, 94),   # 인공나지
    "710": (100, 100, 100, 100),  # 내륙수
}
# 논은 지침에서 AMC별 값을 따로 준다
PADDY = {"I": 70, "II": 79, "III": 89}

SOIL_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

# 선행 5일 강우량(mm) 기준 AMC 구분 (SCS 기준) — (I 상한, III 하한)
AMC_LIMITS = {"growing": (35.6, 53.3), "dormant": (12.7, 27.9)}
GROWING_MONTHS = range(5, 11)   # 5~10월을 성수기로 본다


def convert_cn(cn2, amc):
    cn2 = np.asarray(cn2, dtype=float)
    if amc == "I":
        out = 4.2 * cn2 / (10 - 0.058 * cn2)
    elif amc == "III":
        out = 23 * cn2 / (10 + 0.13 * cn2)
    else:
        out = cn2
    return np.where(cn2 >= 100, 100.0, out)


def cn_table(codes, soil="B", amc="II"):
    """분류 코드 순서대로 CN 배열."""
    s = SOIL_IDX[soil]
    cn2 = np.array([CN_II[c][s] for c in codes], dtype=float)
    cn = convert_cn(cn2, amc)
    for i, c in enumerate(codes):
        if c == "210":
            cn[i] = PADDY[amc]
    return cn


def runoff_mm(P, cn):
    """누적 강수 P(mm) → 직접유출고 Q(mm). P, cn 은 같은 모양으로 브로드캐스트."""
    P = np.asarray(P, dtype=float)
    cn = np.asarray(cn, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(cn >= 100, 0.0, 25400.0 / cn - 254.0)
    Ia = 0.2 * S
    pe = np.maximum(P - Ia, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(pe > 0, pe * pe / (pe + S), 0.0)
    return Q


def amc_from_antecedent(p5, month):
    lo, hi = AMC_LIMITS["growing" if month in GROWING_MONTHS else "dormant"]
    if p5 < lo:
        return "I"
    if p5 > hi:
        return "III"
    return "II"
