"""기상청 레이더 합성자료(HSR) 내려받기와 해독.

자료 형식 (기상청 '레이더 합성자료 포맷 정의서' 기준)
  - 파일: RDR_CMP_HSR_PUB_YYYYMMDDHHMI.bin.gz, 5분 간격, 500 m 격자
  - 머리말 1024 bytes 뒤에 int16 격자 2305(x) × 2881(y)
  - 값/100 = 반사도(dBZ). 왼쪽 아래(남서)에서 시작해 행 단위로 위로 저장
  - -30000: 관측 반경 밖, -25000: 관측 영역 안 무강수, -20000: 표시용 최솟값
  - 투영: LCC(표준위선 30°, 60°, 원점 38°N 126°E), 원점 격자 (1121, 1681)
"""
import datetime as dt
import gzip
import io
import struct
import zipfile

import numpy as np
import requests

from geo import LCC

API_URL = "https://apihub.kma.go.kr/api/typ04/url/rdr_cmp_file.php"

NX, NY = 2305, 2881
HEADER = 1024
DXY = 500.0
CX, CY = 1121, 1681  # 원점(38N, 126E)이 놓인 격자 번호 (왼쪽, 아래 기준)

NULL_OUTSIDE = -30000
NO_ECHO = -25000

# Z = a R^b. 기상청 레이더 강수 추정에서 기본으로 쓰는 Marshall-Palmer 계수
ZR_A, ZR_B = 200.0, 1.6


def _unpack(content):
    """gzip / zip / 압축 안 된 바이너리를 모두 받아준다."""
    if content[:2] == b"\x1f\x8b":
        return gzip.decompress(content)
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            return z.read(z.namelist()[0])
    return content


def download(tm, auth_key, timeout=60):
    """tm(KST, datetime)의 HSR 이진 파일을 받아 압축을 푼 bytes로 돌려준다."""
    params = {
        "tm": tm.strftime("%Y%m%d%H%M"),
        "data": "bin",
        "cmp": "hsr",
        "authKey": auth_key,
    }
    r = requests.get(API_URL, params=params, timeout=timeout)
    # 자료가 없거나 키가 틀리면 압축 파일 대신 짧은 안내문(텍스트)이 온다
    c = r.content
    if r.status_code == 200 and c[:2] in (b"\x1f\x8b", b"PK"):
        return _unpack(c)
    if r.status_code == 200 and len(c) >= HEADER + NX * NY * 2:
        return c
    try:
        msg = c[:300].decode("utf-8").strip()       # 오류 안내(JSON)는 UTF-8
    except UnicodeDecodeError:
        msg = c[:300].decode("euc-kr", "replace").strip()
    raise RuntimeError(f"레이더 자료 없음 ({params['tm']}, HTTP {r.status_code}): {msg}")


def download_latest(auth_key, now=None, tries=8):
    """가장 최근 자료를 찾는다. 생산까지 몇 분 걸리므로 5분씩 거슬러 올라간다."""
    now = now or dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).replace(tzinfo=None)
    tm = now.replace(second=0, microsecond=0)
    tm -= dt.timedelta(minutes=tm.minute % 5)
    last_err = None
    for _ in range(tries):
        try:
            return tm, download(tm, auth_key)
        except Exception as e:  # noqa: BLE001
            last_err = e
            tm -= dt.timedelta(minutes=5)
    raise RuntimeError(f"최근 {tries * 5}분 안에 받을 수 있는 레이더 자료가 없습니다: {last_err}")


def decode(raw):
    """bytes → (ny, nx) int16 배열. 행 0이 가장 남쪽."""
    for endian in ("<", ">"):
        nx, ny = struct.unpack_from(endian + "hh", raw, 20)
        if (nx, ny) == (NX, NY):
            break
    else:
        raise ValueError(f"예상과 다른 격자 크기입니다: {nx}×{ny}")
    data = np.frombuffer(raw, dtype=endian + "i2", count=NX * NY, offset=HEADER)
    return data.reshape(NY, NX)


def dbz_to_rain(dbz):
    return (10.0 ** (dbz / 10.0) / ZR_A) ** (1.0 / ZR_B)


def to_rain_rate(vals):
    """int16 원자료 → 강수강도(mm/h). 관측 밖은 NaN, 무강수는 0."""
    vals = np.asarray(vals)
    rain = np.zeros(vals.shape, dtype=np.float32)
    valid = vals > -20000
    rain[valid] = dbz_to_rain(vals[valid] / 100.0)
    rain[vals <= NULL_OUTSIDE] = np.nan
    # 5 dBZ(약 0.04 mm/h) 미만은 잡음으로 보고 0 처리
    rain[valid & (vals < 500)] = 0.0
    return rain


_lcc = LCC()


def lonlat_to_index(lon, lat):
    """경위도 → 원자료 배열 인덱스(행: 남쪽부터, 열: 서쪽부터)."""
    x, y = _lcc.forward(lon, lat)
    col = np.rint(CX + x / DXY).astype(int)
    row = np.rint(CY + y / DXY).astype(int)
    return row, col


def sample_to_lonlat_grid(grid, lon_min, lat_min, lon_max, lat_max, step):
    """원자료 격자에서 위경도 등간격 격자로 다시 뽑는다(최근접).

    돌려주는 배열은 행 0이 북쪽이라 그대로 PNG로 저장하면 된다.
    범위는 웹 지도 L.imageOverlay 에 그대로 넣을 수 있는 [남서, 북동] 모서리다.
    """
    lons = np.arange(lon_min + step / 2, lon_max, step)
    lats = np.arange(lat_max - step / 2, lat_min, -step)
    LON, LAT = np.meshgrid(lons, lats)
    row, col = lonlat_to_index(LON, LAT)
    ok = (row >= 0) & (row < grid.shape[0]) & (col >= 0) & (col < grid.shape[1])
    out = np.full(LON.shape, NULL_OUTSIDE, dtype=grid.dtype)
    out[ok] = grid[row[ok], col[ok]]
    bounds = [
        [lat_max - step * len(lats), lon_min],
        [lat_max, lon_min + step * len(lons)],
    ]
    return out, LON, LAT, bounds
