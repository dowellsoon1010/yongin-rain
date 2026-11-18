"""용인시 실시간 강수·유출 지도 — 자료 수집·가공 스크립트.

  python pipeline/fetch.py            # 실제 기상청 자료 (환경변수 KMA_API_KEY 필요)
  python pipeline/fetch.py --demo     # 가짜 자료로 전체 흐름 시험 (키 불필요)
  python pipeline/fetch.py --gis      # QGIS용 격자 파일(rain.asc, runoff.asc)도 저장

하는 일
  1) 기상청 레이더(5분 간격)를 최근 4시간치 모은다 (이미 받은 장면은 state/에 보관)
  2) 장면마다 최근 3시간 누적 강수 P를 구하고, 토지피복별 CN으로 직접유출 Q를 계산
  3) 최근 1시간(13장면)의 강수·유출 그림과 구별·토지피복별 통계를 web/data/에 저장

준비물: pipeline/landcover_grid.npz (prepare_landcover.py 로 한 번 만든다)
"""
import argparse
import datetime as dt
import json
import os
import shutil
import struct
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import aws  # noqa: E402
import radar  # noqa: E402
import runoff  # noqa: E402
from geo import cell_area_m2, load_districts, which_district  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "web", "data")
FRAMES = os.path.join(OUT, "frames")
STATE = os.path.join(ROOT, "state")            # 실행 사이에 이어지는 자료 (git에는 안 올림)
RAIN_CACHE = os.path.join(STATE, "rain")
LC_PATH = os.path.join(ROOT, "pipeline", "landcover_grid.npz")

# ---- 설정 -----------------------------------------------------------------
DISTRICTS = os.path.join(OUT, "districts.geojson")   # 지도 테두리, AWS 지점 구 판정용
STATIONS_CSV = os.path.join(ROOT, "pipeline", "stations.csv")
SOIL_GROUP = os.environ.get("SOIL_GROUP", "B")  # 수문학적 토양군 (A~D). 토양도가 없어 하나로 가정
COARSE = 0.002       # 레이더 그림 격자(도) ≈ 180 m × 220 m
MARGIN_CELLS = 75    # 용인 바깥 여백 칸 수 (0.15° ≈ 13~17 km). 다가오는 비구름도 보이게
STN_BUFFER = 0.03
FRAME_STEP = 5       # 레이더 간격(분)
PLAY = 13            # 재생 장면 수 (최근 1시간)
WINDOW = 36          # 누적 강수 구간: 36 × 5분 = 3시간
MAX_FILL = 3         # 빠진 장면은 앞 장면 값으로 채움 (최대 3장 = 15분)
HISTORY_LEN = 36
WET = 0.1

RAIN_COLORS = [
    (0.1, (166, 227, 255)), (0.5, (94, 196, 255)), (1, (42, 140, 255)),
    (2, (34, 197, 94)), (5, (155, 225, 93)), (10, (255, 225, 77)),
    (20, (255, 162, 58)), (30, (255, 90, 60)), (50, (214, 42, 107)),
    (70, (155, 48, 217)), (90, (75, 31, 168)),
]
# 3시간 누적 유출고(mm)
RUNOFF_COLORS = [
    (0.5, (208, 231, 255)), (1, (166, 207, 255)), (2, (116, 176, 255)),
    (5, (61, 139, 253)), (10, (31, 95, 214)), (20, (27, 63, 160)),
    (30, (74, 42, 154)), (50, (123, 31, 162)), (80, (194, 24, 91)),
]
GROUP_ORDER = ["시가화건조지역", "농업지역", "산림지역", "초지", "습지", "나지", "수역"]
GROUP_COLORS = {"시가화건조지역": "#e74c3c", "농업지역": "#f1c40f", "산림지역": "#1e8449",
                "초지": "#a9dc76", "습지": "#9b59b6", "나지": "#85c1e9", "수역": "#2e86c1"}

KST = dt.timezone(dt.timedelta(hours=9))


def intensity_label(r):
    """기상청 강수강도 구분"""
    if r is None or r < WET:
        return "없음"
    if r < 3:
        return "약한 비"
    if r < 15:
        return "보통 비"
    if r < 30:
        return "강한 비"
    return "매우 강한 비"


def r1(x, d=1):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


# ---- 격자 -----------------------------------------------------------------
class Grids:
    """토지피복 격자(약 25 m)와, 그것을 8×8로 묶은 레이더 그림 격자(0.002°)."""

    def __init__(self):
        if not os.path.exists(LC_PATH):
            sys.exit("pipeline/landcover_grid.npz 가 없습니다. prepare_landcover.py 를 먼저 실행하세요.")
        z = np.load(LC_PATH)
        self.cls = z["cls"]
        self.dist = z["dist"]
        self.sub = int(z["sub"])
        self.fstep = float(z["step"])
        self.codes = [str(c) for c in z["codes"]]
        self.names = [str(c) for c in z["names"]]
        self.groups = [str(c) for c in z["groups"]]
        self.dnames = [str(c) for c in z["dist_names"]]
        self.lc_lon0 = float(z["lon_min"])
        self.lc_lat1 = float(z["lat_max"])
        fy, fx = self.cls.shape
        self.cy, self.cx = fy // self.sub, fx // self.sub  # 토지피복 범위의 굵은 칸 수
        self.lc_bounds = [[self.lc_lat1 - fy * self.fstep, self.lc_lon0],
                          [self.lc_lat1, self.lc_lon0 + fx * self.fstep]]

        # 레이더 그림 격자 = 토지피복 범위 + 사방 여백
        m = MARGIN_CELLS
        self.lon0 = self.lc_lon0 - m * COARSE
        self.lat1 = self.lc_lat1 + m * COARSE
        self.ny, self.nx = self.cy + 2 * m, self.cx + 2 * m
        self.bounds = [[self.lat1 - self.ny * COARSE, self.lon0],
                       [self.lat1, self.lon0 + self.nx * COARSE]]
        lons = self.lon0 + (np.arange(self.nx) + 0.5) * COARSE
        lats = self.lat1 - (np.arange(self.ny) + 0.5) * COARSE
        LON, LAT = np.meshgrid(lons, lats)
        self.rrow, self.rcol = radar.lonlat_to_index(LON, LAT)

        # 굵은 칸별 (구, 분류) 면적 m² : A[j, k, y, x]
        flats = self.lc_lat1 - (np.arange(fy) + 0.5) * self.fstep
        farea = np.repeat(cell_area_m2(flats, self.fstep)[:, None], fx, axis=1)
        nk, nd = len(self.codes), len(self.dnames)
        idx = (np.where(self.dist < nd, self.dist, nd).astype(np.int64) * (nk + 1)
               + np.where(self.cls < nk, self.cls, nk))
        yy = np.arange(fy)[:, None] // self.sub
        xx = np.arange(fx)[None, :] // self.sub
        cell = (yy * self.cx + xx)
        flat = idx * (self.cy * self.cx) + cell
        A = np.bincount(flat.ravel(), weights=farea.ravel(),
                        minlength=(nd + 1) * (nk + 1) * self.cy * self.cx)
        self.A = A.reshape(nd + 1, nk + 1, self.cy, self.cx)[:nd, :nk]
        # 고운 격자에서 굵은 칸 번호 (유출 그림 만들 때)
        self.fy_idx = np.broadcast_to(yy, (fy, fx))
        self.fx_idx = np.broadcast_to(xx, (fy, fx))

    def lc_window(self, g):
        m = MARGIN_CELLS
        return g[m:m + self.cy, m:m + self.cx]

    def sample(self, raw_grid):
        ok = ((self.rrow >= 0) & (self.rrow < raw_grid.shape[0])
              & (self.rcol >= 0) & (self.rcol < raw_grid.shape[1]))
        out = np.full(self.rrow.shape, radar.NULL_OUTSIDE, dtype=raw_grid.dtype)
        out[ok] = raw_grid[self.rrow[ok], self.rcol[ok]]
        return radar.to_rain_rate(out)


# ---- 그림 -----------------------------------------------------------------
def render(values, colors, path, nan_rgba=None, alpha=215):
    h, w = values.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    v = np.nan_to_num(values, nan=-1)
    for lo, rgb in colors:
        m = v >= lo
        rgba[m, :3] = rgb
        rgba[m, 3] = alpha
    if nan_rgba is not None:
        rgba[np.isnan(values)] = nan_rgba
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def write_asc(values, bounds, step, path, digits=2):
    (south, west), _ = bounds
    h, w = values.shape
    vals = np.where(np.isnan(values), -9999, np.round(values, digits))
    with open(path, "w") as f:
        f.write(f"ncols {w}\nnrows {h}\nxllcorner {west}\nyllcorner {south}\n"
                f"cellsize {step}\nNODATA_value -9999\n")
        np.savetxt(f, vals, fmt="%g")


# ---- 레이더 보관함 ---------------------------------------------------------
def cache_path(t):
    return os.path.join(RAIN_CACHE, f"{t:%Y%m%d%H%M}.npz")


def load_cached(t, shape=None):
    try:
        r = np.load(cache_path(t))["r"]
    except (OSError, KeyError, ValueError):
        return None
    if shape is not None and r.shape != tuple(shape):
        return None  # 지도 범위를 바꾸기 전에 저장한 자료 → 다시 받는다
    out = r.astype(np.float32) / 100.0
    out[r == 65535] = np.nan
    return out


def save_cached(t, rate):
    r = np.where(np.isnan(rate), 65535, np.clip(np.rint(rate * 100), 0, 65534)).astype(np.uint16)
    np.savez_compressed(cache_path(t), r=r)


def prune_cache(keep_from):
    for fn in os.listdir(RAIN_CACHE):
        try:
            t = dt.datetime.strptime(fn[:12], "%Y%m%d%H%M")
        except ValueError:
            continue
        if t < keep_from:
            os.remove(os.path.join(RAIN_CACHE, fn))


# ---- 데모 자료 -------------------------------------------------------------
_demo_xy = None


def demo_radar_bytes(tm):
    """실제 파일과 같은 형식의 가짜 HSR 파일 (해독 과정까지 시험하려고)."""
    global _demo_xy
    if _demo_xy is None:
        rows, cols = np.mgrid[0:radar.NY, 0:radar.NX]
        _demo_xy = ((cols - radar.CX) * radar.DXY, (rows - radar.CY) * radar.DXY)
    x, y = _demo_xy
    dbz = np.full(x.shape, -250.0, dtype=np.float32)
    k0 = (tm.hour * 60 + tm.minute) / 5.0
    k = k0 % 96   # 8시간마다 처음부터 반복
    # 남서→북동으로 지나가는 비구름 띠와 소나기 세포 (경도, 위도, 반경 m, 최대 dBZ)
    cells = [
        (126.80 + 0.0065 * k, 37.12 + 0.0012 * k, 15_000, 46),
        (126.86 + 0.0060 * k, 37.22 + 0.0006 * k, 8_000, 50),
        (126.72 + 0.0070 * k, 37.30 + 0.0004 * k, 22_000, 38),
    ]
    for lon, lat, rad, peak in cells:
        cx, cy = radar._lcc.forward(lon, lat)
        r2 = ((x - cx) ** 2 + (y - cy) ** 2) / rad ** 2
        near = r2 < 1.3
        cell = peak - 28 * r2 + 3 * np.sin(x / 2500 + k0) * np.cos(y / 3100)
        dbz = np.where(near, np.maximum(dbz, cell), dbz)
    vals = np.rint(dbz * 100).astype("<i2")
    vals[:200, :] = radar.NULL_OUTSIDE
    head = bytearray(radar.HEADER)
    struct.pack_into("<hhh", head, 20, radar.NX, radar.NY, 1)
    return bytes(head) + vals.tobytes()


def demo_stations(districts):
    rng = np.random.default_rng(7)
    pts = np.vstack([r for d in districts for r in d["rings"]])
    lon0, lat0 = pts.min(0)
    lon1, lat1 = pts.max(0)
    out = []
    while len(out) < 10:
        lon, lat = rng.uniform(lon0, lon1), rng.uniform(lat0, lat1)
        if which_district(lon, lat, districts):
            out.append({"stn": str(900 + len(out)), "name": f"데모지점{len(out) + 1}", "lon": lon, "lat": lat})
    return out


def demo_minute(stations, grids, rate, p1h, tm):
    rng = np.random.default_rng(int(tm.timestamp()) // 300)
    out = {}
    for s in stations:
        c = int((s["lon"] - grids.lon0) / COARSE)
        r = int((grids.lat1 - s["lat"]) / COARSE)
        now = float(np.nan_to_num(rate[r, c])) * rng.uniform(0.8, 1.2)
        hr = float(np.nan_to_num(p1h[r, c])) * rng.uniform(0.8, 1.2)
        out[s["stn"]] = {
            "time": tm.strftime("%Y%m%d%H%M"),
            "rn15": round(now / 4, 1), "rn60": round(hr, 1), "rnday": round(hr * 2.2, 1),
            "rain_now": now > 0.1, "ta": round(rng.uniform(16, 20), 1), "hm": round(rng.uniform(85, 98)),
            "ws": round(rng.uniform(1, 5), 1), "wd": round(rng.uniform(180, 260)),
        }
    return out


# ---- 선행강우(AMC) ---------------------------------------------------------
def update_rain_log(log, times, yongin_depth):
    for t, d in zip(times, yongin_depth):
        if d is not None:
            log[t.strftime("%Y%m%d%H%M")] = round(float(d), 3)
    cutoff = (times[-1] - dt.timedelta(days=6)).strftime("%Y%m%d%H%M")
    return {k: v for k, v in sorted(log.items()) if k >= cutoff}


def decide_amc(log, tm):
    """사건(최근 3시간) 이전 5일 강수로 AMC 결정. 기록이 5일치가 안 되면 AMC-II."""
    end = tm - dt.timedelta(minutes=FRAME_STEP * WINDOW)
    start = end - dt.timedelta(days=5)
    keys = [k for k in log if start.strftime("%Y%m%d%H%M") <= k < end.strftime("%Y%m%d%H%M")]
    need = 5 * 24 * 60 / FRAME_STEP
    coverage = len(keys) / need
    p5 = sum(log[k] for k in keys)
    if coverage < 0.9:
        return "II", round(p5, 1), round(coverage * 100), "선행 5일 기록이 아직 부족해 보통 조건(AMC-II)을 씀"
    amc = runoff.amc_from_antecedent(p5, tm.month)
    return amc, round(p5, 1), round(coverage * 100), f"선행 5일 강수 {p5:.1f} mm 기준"


# ---- 본체 -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="가짜 자료로 시험")
    ap.add_argument("--time", help="과거 시각 재현 (KST, YYYYMMDDHHMM)")
    ap.add_argument("--gis", action="store_true", help="QGIS용 격자 파일도 저장")
    args = ap.parse_args()

    for d in (OUT, RAIN_CACHE):
        os.makedirs(d, exist_ok=True)
    g = Grids()
    districts = load_districts(DISTRICTS, "name")
    nk, nd = len(g.codes), len(g.dnames)

    now = (dt.datetime.strptime(args.time, "%Y%m%d%H%M") if args.time
           else dt.datetime.now(KST).replace(tzinfo=None))
    key = None
    if not args.demo:
        key = os.environ.get("KMA_API_KEY")
        if not key:
            sys.exit("환경변수 KMA_API_KEY 에 기상청 API허브 인증키를 넣어주세요.")

    def fetch_rate(t):
        cached = load_cached(t, (g.ny, g.nx))
        if cached is not None:
            return cached
        raw = demo_radar_bytes(t) if args.demo else radar.download(t, key)
        rate = g.sample(radar.decode(raw))
        save_cached(t, rate)
        return rate

    # 1) 최신 시각
    if args.demo or args.time:
        tm = now.replace(second=0, microsecond=0) - dt.timedelta(minutes=now.minute % 5)
    else:
        tm, raw = radar.download_latest(key, now)
        if load_cached(tm, (g.ny, g.nx)) is None:
            save_cached(tm, g.sample(radar.decode(raw)))

    # 2) 최근 4시간(누적 3시간 + 재생 1시간) 장면 모으기
    n_all = PLAY + WINDOW - 1
    times = [tm - dt.timedelta(minutes=FRAME_STEP * k) for k in range(n_all - 1, -1, -1)]
    rates, missing = [], []
    for t in times:
        try:
            rates.append(fetch_rate(t))
        except Exception as e:  # noqa: BLE001
            print(f"{t:%H:%M} 레이더 없음: {e}", file=sys.stderr)
            rates.append(None)
            missing.append(t.strftime("%H:%M"))
    if rates[-1] is None:
        raise RuntimeError("최신 레이더 장면이 없습니다.")
    prune_cache(times[0] - dt.timedelta(minutes=30))

    # 빠진 장면은 앞 장면 값으로 (최대 15분), 그보다 길면 0
    filled, last, gap = [], None, 0
    for r in rates:
        if r is not None:
            last, gap = r, 0
            filled.append(r)
        else:
            gap += 1
            filled.append(last if (last is not None and gap <= MAX_FILL) else None)
    zeros = np.zeros((g.ny, g.nx), np.float32)
    rate_stack = np.stack([zeros if r is None else r for r in filled])         # mm/h
    depth = np.nan_to_num(rate_stack) * (FRAME_STEP / 60.0)                      # 5분 강수량 mm
    depth_lc = depth[:, MARGIN_CELLS:MARGIN_CELLS + g.cy, MARGIN_CELLS:MARGIN_CELLS + g.cx]

    # 3) 선행강우 → AMC → 분류별 CN
    log_path = os.path.join(STATE, "rain_log.json")
    try:
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    except (OSError, ValueError):
        log = {}
    A_tot = g.A.sum(axis=(0, 1))
    yongin_depth = [None if r is None else float((depth_lc[i] * A_tot).sum() / A_tot.sum())
                    for i, r in enumerate(rates)]
    if not args.demo:
        log = update_rain_log(log, times, yongin_depth)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f)
    amc, p5, cover, amc_note = decide_amc(log, tm) if not args.demo else ("II", None, 0, "데모: 보통 조건(AMC-II)")
    cn = runoff.cn_table(g.codes, SOIL_GROUP, amc)

    # 4) 재생 장면별 계산
    # 폴더째 지우지 않고 안의 그림만 지운다 (OneDrive 등이 폴더를 잡고 있어도 되게)
    os.makedirs(FRAMES, exist_ok=True)
    for fn in os.listdir(FRAMES):
        if fn.endswith(".png"):
            try:
                os.remove(os.path.join(FRAMES, fn))
            except OSError:
                pass  # 지우지 못한 옛 그림은 남아도 지도에서 쓰지 않는다
    A = g.A                                   # (nd, nk, cy, cx) m²
    A_d = A.sum(axis=1)                       # 구별 칸 면적
    area_d = A_d.sum(axis=(1, 2))
    group_idx = {gname: [k for k in range(nk) if g.groups[k] == gname] for gname in GROUP_ORDER}
    cls_safe = np.where(g.cls < nk, g.cls, 0)
    frames = []
    for j in range(WINDOW - 1, n_all):
        t = times[j]
        P = depth_lc[j - WINDOW + 1:j + 1].sum(axis=0)          # 3시간 누적 (cy, cx)
        P_prev = P - depth_lc[j]
        Qk = runoff.runoff_mm(P[None], cn[:, None, None])       # (nk, cy, cx)
        Qk_prev = runoff.runoff_mm(P_prev[None], cn[:, None, None])
        vol = np.einsum("dkyx,kyx->dk", A, Qk) / 1000.0           # m³
        vol_prev = np.einsum("dkyx,kyx->dk", A, Qk_prev) / 1000.0
        rain_vol = np.einsum("dyx,yx->d", A_d, P) / 1000.0
        rate_lc = g.lc_window(rate_stack[j])
        stats = []
        for d in range(nd):
            m = A_d[d] > 0
            rr = rate_lc[m]
            w = A_d[d][m]
            ok = ~np.isnan(rr)
            mean_rate = float((rr[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else None
            max_rate = float(rr[ok].max()) if ok.any() else None
            wet = float(w[ok & (rr >= WET)].sum() / w[ok].sum() * 100) if ok.any() else None
            qv = vol[d].sum()
            groups = []
            for gname in GROUP_ORDER:
                ks = group_idx[gname]
                ga = float(A[d, ks].sum())
                gv = float(vol[d, ks].sum())
                groups.append({"g": gname, "area_pct": r1(ga / area_d[d] * 100),
                               "q_pct": r1(gv / qv * 100) if qv > 0 else None,
                               "q_mm": r1(gv / ga * 1000 if ga > 0 else 0, 2)})
            stats.append({
                "name": g.dnames[d],
                "mean": r1(mean_rate, 2), "max": r1(max_rate), "wet_pct": r1(wet),
                "label": intensity_label(max_rate),
                "p3h": r1(rain_vol[d] / area_d[d] * 1000),
                "q3h": r1(qv / area_d[d] * 1000, 2),
                "q_m3": int(round(qv)),
                "ratio": r1(qv / rain_vol[d] * 100) if rain_vol[d] > 0 else None,
                "q_rate": r1((qv - vol_prev[d].sum()) / area_d[d] * 1000 * 60 / FRAME_STEP, 2),
                "groups": groups,
            })

        # 그림: 강수(넓은 범위, 굵은 격자) / 유출(용인, 고운 격자)
        tag = f"{t:%Y%m%d%H%M}"
        render(rate_stack[j], RAIN_COLORS, os.path.join(FRAMES, f"rain_{tag}.png"), nan_rgba=(120, 120, 120, 60))
        Qf = Qk[cls_safe, g.fy_idx, g.fx_idx]
        Qf = np.where(g.cls < nk, Qf, np.nan)
        render(Qf, RUNOFF_COLORS, os.path.join(FRAMES, f"runoff_{tag}.png"), alpha=230)
        city_p = float((P * A_tot).sum() / A_tot.sum())
        frames.append({"t": t.strftime("%Y-%m-%d %H:%M"), "rain": f"frames/rain_{tag}.png",
                       "runoff": f"frames/runoff_{tag}.png", "p3h_city": r1(city_p),
                       "filled": rates[j] is None, "districts": stats})
        if j == n_all - 1:
            last_P, last_Qf, last_P1h = P, Qf, depth[j - 11:j + 1].sum(axis=0)

    if args.gis:
        write_asc(rate_stack[-1], g.bounds, COARSE, os.path.join(OUT, "rain.asc"))
        write_asc(last_Qf, g.lc_bounds, g.fstep, os.path.join(OUT, "runoff.asc"))
        write_asc(last_P, g.lc_bounds, COARSE, os.path.join(OUT, "rain3h.asc"))

    # 5) AWS (최신 시각)
    stations, obs, aws_err = [], {}, None
    try:
        if args.demo:
            stations = demo_stations(districts)
            obs = demo_minute(stations, g, rate_stack[-1], last_P1h, tm)
        else:
            stations = (aws.load_stations_csv(STATIONS_CSV) if os.path.exists(STATIONS_CSV)
                        else aws.fetch_stations(key))
            obs = aws.fetch_minute(key, None if not args.time else now)
    except Exception as e:  # noqa: BLE001
        aws_err = str(e)
        print("AWS 자료 실패:", e, file=sys.stderr)
    (s0, w0), (n0, e0) = g.lc_bounds
    points = []
    for s in stations:
        if not (w0 - STN_BUFFER <= s["lon"] <= e0 + STN_BUFFER and s0 - STN_BUFFER <= s["lat"] <= n0 + STN_BUFFER):
            continue
        o = obs.get(s["stn"])
        if o is None:
            continue
        points.append({**s, **o, "district": which_district(s["lon"], s["lat"], districts),
                       "time": aws.kst_time(o.get("time"))})
    latest_stats = [dict(s) for s in frames[-1]["districts"]]
    for st in latest_stats:
        inside = [p["rn60"] for p in points if p["district"] == st["name"] and p["rn60"] is not None]
        st["aws_n"] = len(inside)
        st["aws_rn60_mean"] = round(sum(inside) / len(inside), 1) if inside else None

    # 6) 저장
    latest = {
        "demo": bool(args.demo),
        "replay": bool(args.time),
        "radar_time": tm.strftime("%Y-%m-%d %H:%M"),
        "generated": dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "bounds": g.bounds, "lc_bounds": g.lc_bounds,
        "districts": latest_stats, "frames": frames, "stations": points,
        "aws_error": aws_err, "missing": missing,
        "zr": f"Z = {radar.ZR_A:g} R^{radar.ZR_B:g}",
        "model": {
            "method": "SCS-CN", "window_h": WINDOW * FRAME_STEP / 60, "soil": SOIL_GROUP,
            "amc": amc, "amc_note": amc_note, "p5": p5, "p5_coverage": cover,
            "classes": [{"code": c, "name": n, "group": gr, "cn": round(float(v), 1)}
                        for c, n, gr, v in zip(g.codes, g.names, g.groups, cn)],
            "groups": [{"g": gname, "color": GROUP_COLORS[gname]} for gname in GROUP_ORDER],
            "rain_colors": [[lo, "#%02x%02x%02x" % c] for lo, c in RAIN_COLORS],
            "runoff_colors": [[lo, "#%02x%02x%02x" % c] for lo, c in RUNOFF_COLORS],
        },
    }
    with open(os.path.join(OUT, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(latest, f, ensure_ascii=False, indent=1)
    shutil.copyfile(os.path.join(OUT, frames[-1]["rain"]), os.path.join(OUT, "radar.png"))

    hpath = os.path.join(STATE, "history.json")
    try:
        with open(hpath, encoding="utf-8") as f:
            hist = {h["t"]: h for h in json.load(f) if h.get("demo") == latest["demo"]}
    except (OSError, ValueError):
        hist = {}
    for fr in frames:
        hist[fr["t"]] = {"t": fr["t"], "demo": latest["demo"],
                         "v": {s["name"]: s["mean"] for s in fr["districts"]},
                         "q": {s["name"]: s["q3h"] for s in fr["districts"]}}
    hist = sorted(hist.values(), key=lambda h: h["t"])[-HISTORY_LEN:]
    with open(hpath, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False)
    shutil.copyfile(hpath, os.path.join(OUT, "history.json"))

    s = latest_stats
    print(f"[{latest['radar_time']}] AMC-{amc}, 토양군 {SOIL_GROUP} | " + ", ".join(
        f"{x['name']} 3시간 강수 {x['p3h']} mm → 유출 {x['q3h']} mm({x['ratio']}%)" for x in s)
        + f" | AWS {len(points)}곳" + (f" | 빠진 장면 {len(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
