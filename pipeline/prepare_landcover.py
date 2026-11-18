"""토지피복도 → 격자 변환 (한 번만 실행하는 준비 스크립트).

  python pipeline/prepare_landcover.py landcover.geojson

입력: QGIS에서 EPSG:4326 GeoJSON으로 내보낸 환경부 토지피복도(중분류 L2_CODE,
      읍면동 코드 ADM_CD 포함)
출력:
  pipeline/landcover_grid.npz   약 25 m 격자의 피복 분류·구 번호 (실시간 계산용)
  web/data/landcover.png        토지피복 지도 그림
  web/data/landcover.json       그림 범위, 분류별 색·면적, 구별 면적
  web/data/districts.geojson    격자에서 뽑은 구 경계 (지도 테두리용)

실시간 스크립트(fetch.py)는 numpy만 쓰지만, 이 준비 스크립트는 다각형 판정에
matplotlib 을 쓴다 (pip install matplotlib).
"""
import json
import math
import os
import sys
import time

import numpy as np
from matplotlib.path import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo import cell_area_m2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_NPZ = os.path.join(ROOT, "pipeline", "landcover_grid.npz")
OUT_WEB = os.path.join(ROOT, "web", "data")

COARSE = 0.002   # 레이더 그림 격자(도)와 맞춘다
SUB = 8          # 한 칸을 8×8로 나눔 → 0.00025° ≈ 22 m(경도) × 28 m(위도)
STEP = COARSE / SUB

# 구 코드(센서스 코드 앞 5자리)
DISTRICTS = {"31193": "수지구", "31192": "기흥구", "31191": "처인구"}

# 환경부 토지피복 중분류: 코드, 이름, 대분류, 색
CLASSES = [
    ("110", "주거지역", "시가화건조지역", (231, 76, 60)),
    ("120", "공업지역", "시가화건조지역", (176, 58, 46)),
    ("130", "상업지역", "시가화건조지역", (255, 99, 132)),
    ("140", "문화·체육·휴양지역", "시가화건조지역", (240, 128, 128)),
    ("150", "교통지역", "시가화건조지역", (120, 40, 31)),
    ("160", "공공시설지역", "시가화건조지역", (205, 97, 85)),
    ("210", "논", "농업지역", (244, 208, 63)),
    ("220", "밭", "농업지역", (247, 220, 111)),
    ("230", "시설재배지", "농업지역", (214, 162, 32)),
    ("240", "과수원", "농업지역", (230, 190, 90)),
    ("250", "기타재배지", "농업지역", (250, 232, 160)),
    ("310", "활엽수림", "산림지역", (39, 174, 96)),
    ("320", "침엽수림", "산림지역", (20, 110, 60)),
    ("330", "혼효림", "산림지역", (30, 140, 75)),
    ("410", "자연초지", "초지", (171, 235, 120)),
    ("420", "인공초지", "초지", (200, 240, 160)),
    ("510", "내륙습지", "습지", (155, 89, 182)),
    ("610", "자연나지", "나지", (133, 193, 233)),
    ("620", "인공나지", "나지", (174, 214, 241)),
    ("710", "내륙수", "수역", (41, 128, 185)),
]
CODE_IDX = {c[0]: i for i, c in enumerate(CLASSES)}
NONE = 255


def simplify(ring, tol):
    """Douglas-Peucker 단순화 (tol: 도 단위)."""
    keep = np.zeros(len(ring), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = ring[i], ring[j]
        seg = b - a
        L = np.hypot(*seg)
        pts = ring[i + 1:j]
        d = (np.abs(seg[0] * (pts[:, 1] - a[1]) - seg[1] * (pts[:, 0] - a[0])) / L
             if L > 0 else np.hypot(*(pts - a).T))
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return ring[keep]


def rings_of(geom):
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [[np.asarray(r, dtype=float)[:, :2] for r in p] for p in polys]


def main(src):
    t0 = time.time()
    with open(src, encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"]
    print(f"피처 {len(feats):,}개 읽음 ({time.time() - t0:.0f}초)")

    allpts = np.vstack([r for f in feats for p in rings_of(f["geometry"]) for r in p])
    lon_min = math.floor(allpts[:, 0].min() / COARSE) * COARSE
    lat_min = math.floor(allpts[:, 1].min() / COARSE) * COARSE
    lon_max = math.ceil(allpts[:, 0].max() / COARSE) * COARSE
    lat_max = math.ceil(allpts[:, 1].max() / COARSE) * COARSE
    nx = int(round((lon_max - lon_min) / STEP))
    ny = int(round((lat_max - lat_min) / STEP))
    print(f"격자 {nx}×{ny} (칸 {STEP}°)")

    cls = np.full((ny, nx), NONE, np.uint8)   # 행 0 = 북쪽
    dist = np.full((ny, nx), NONE, np.uint8)
    dkeys = list(DISTRICTS)
    unknown = set()

    for n, f in enumerate(feats):
        p = f["properties"]
        k = CODE_IDX.get(str(p.get("L2_CODE")))
        if k is None:
            unknown.add(p.get("L2_CODE"))
            continue
        dcode = str(p.get("ADM_CD", ""))[:5]
        d = dkeys.index(dcode) if dcode in DISTRICTS else NONE
        for poly in rings_of(f["geometry"]):
            ext = poly[0]
            c0 = max(int((ext[:, 0].min() - lon_min) / STEP), 0)
            c1 = min(int((ext[:, 0].max() - lon_min) / STEP) + 1, nx)
            r0 = max(int((lat_max - ext[:, 1].max()) / STEP), 0)
            r1 = min(int((lat_max - ext[:, 1].min()) / STEP) + 1, ny)
            if c1 <= c0 or r1 <= r0:
                continue
            xs = lon_min + (np.arange(c0, c1) + 0.5) * STEP
            ys = lat_max - (np.arange(r0, r1) + 0.5) * STEP
            X, Y = np.meshgrid(xs, ys)
            pts = np.column_stack([X.ravel(), Y.ravel()])
            inside = np.zeros(len(pts), bool)
            for ring in poly:  # 바깥 경계 XOR 구멍
                inside ^= Path(ring).contains_points(pts)
            if not inside.any():
                continue
            inside = inside.reshape(X.shape)
            cls[r0:r1, c0:c1][inside] = k
            dist[r0:r1, c0:c1][inside] = d
        if n % 10000 == 0:
            print(f"  {n:,}/{len(feats):,} ({time.time() - t0:.0f}초)")
    if unknown:
        print("표에 없는 L2_CODE (건너뜀):", unknown)

    np.savez_compressed(
        OUT_NPZ, cls=cls, dist=dist, lon_min=lon_min, lat_max=lat_max, step=STEP, sub=SUB,
        codes=np.array([c[0] for c in CLASSES]), names=np.array([c[1] for c in CLASSES]),
        groups=np.array([c[2] for c in CLASSES]),
        dist_codes=np.array(dkeys), dist_names=np.array([DISTRICTS[k] for k in dkeys]),
    )

    lats = lat_max - (np.arange(ny) + 0.5) * STEP
    area = np.repeat(cell_area_m2(lats, STEP)[:, None], nx, axis=1)

    # 토지피복 그림
    rgba = np.zeros((ny, nx, 4), np.uint8)
    for i, c in enumerate(CLASSES):
        m = cls == i
        rgba[m, :3] = c[3]
        rgba[m, 3] = 255
    os.makedirs(OUT_WEB, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(os.path.join(OUT_WEB, "landcover.png"), optimize=True)

    # 면적 통계
    stats = {"bounds": [[lat_min, lon_min], [lat_max, lon_max]], "classes": [], "districts": []}
    for i, c in enumerate(CLASSES):
        stats["classes"].append({"code": c[0], "name": c[1], "group": c[2],
                                 "color": "#%02x%02x%02x" % c[3],
                                 "km2": round(float(area[cls == i].sum()) / 1e6, 2)})
    for j, k in enumerate(dkeys):
        m = dist == j
        stats["districts"].append({
            "name": DISTRICTS[k], "km2": round(float(area[m].sum()) / 1e6, 2),
            "class_km2": [round(float(area[m & (cls == i)].sum()) / 1e6, 3) for i in range(len(CLASSES))],
        })
    with open(os.path.join(OUT_WEB, "landcover.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)

    write_district_outlines(dist, dkeys, lon_min, lat_max)
    print(f"완료 ({time.time() - t0:.0f}초)")
    for d in stats["districts"]:
        print(f"  {d['name']} {d['km2']} km²")


def write_district_outlines(dist, dkeys, lon_min, lat_max):
    """격자 구 번호에서 경계선을 뽑아 GeoJSON 으로 저장 (지도 테두리용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ny, nx = dist.shape
    xs = lon_min + (np.arange(nx) + 0.5) * STEP
    ys = lat_max - (np.arange(ny) + 0.5) * STEP
    feats = []
    for j, k in enumerate(dkeys):
        mask = np.pad((dist == j).astype(float), 1)
        X = np.concatenate([[xs[0] - STEP], xs, [xs[-1] + STEP]])
        Y = np.concatenate([[ys[0] + STEP], ys, [ys[-1] - STEP]])
        cs = plt.contour(X, Y, mask, levels=[0.5])
        polys = []
        for path in cs.get_paths():
            for ring in path.to_polygons():
                if len(ring) < 20:  # 아주 작은 조각(경계 잡음)은 버림
                    continue
                polys.append([np.round(simplify(ring, STEP / 4), 6).tolist()])
        plt.close("all")
        # 가장 큰 고리를 바깥 경계로, 나머지는 별도 조각으로 둔다
        polys.sort(key=lambda p: -len(p[0]))
        feats.append({"type": "Feature", "properties": {"name": DISTRICTS[k], "code": k},
                      "geometry": {"type": "MultiPolygon", "coordinates": polys[:3]}})
    with open(os.path.join(OUT_WEB, "districts.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("사용법: python pipeline/prepare_landcover.py landcover.geojson")
    main(sys.argv[1])
