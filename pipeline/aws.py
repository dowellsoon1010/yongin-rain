"""기상청 AWS(방재기상관측) 지점정보와 매분자료.

API허브 텍스트 응답은 '#'으로 시작하는 설명 줄과 자료 줄로 되어 있다.
열 순서가 바뀌어도 깨지지 않도록 머리말의 열 이름을 읽어서 찾는다.
"""
import csv
import datetime as dt

import requests

MIN_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
STN_URL = "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php"

MISSING = -50.0  # 이 값 이하는 결측 (응답 머리말 기준)


def _get_text(url, params, timeout=30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    # API허브 텍스트는 EUC-KR
    return r.content.decode("euc-kr", "replace")


def _split(line):
    line = line.strip().rstrip("=").strip()
    if "," in line:
        return [c.strip() for c in line.split(",")]
    return line.split()


def _parse_table(text, must_have):
    """머리말 열 이름 줄(#로 시작하고 must_have 가 모두 들어있는 줄)을 찾아
    [{열이름: 값}] 으로 바꾼다."""
    header = None
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            names = _split(s.lstrip("#"))
            if all(m in names for m in must_have):
                header = names
            continue
        if header is None:
            continue
        cols = _split(s)
        if len(cols) < len(header) - 1:
            continue
        rows.append(dict(zip(header, cols)))
    if header is None:
        raise ValueError(f"응답에서 열 이름({must_have})을 찾지 못했습니다:\n{text[:500]}")
    return rows


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x <= MISSING else x


# ---------------------------------------------------------------------------
def fetch_stations(auth_key):
    """AWS 지점 목록: [{stn, name, lon, lat}]"""
    text = _get_text(STN_URL, {"inf": "AWS", "stn": "", "help": "1", "authKey": auth_key})
    rows = _parse_table(text, ["LON", "LAT"])
    out = []
    for r in rows:
        stn = r.get("STN_ID") or r.get("STN")
        lon, lat = _num(r.get("LON")), _num(r.get("LAT"))
        if stn is None or lon is None or lat is None:
            continue
        out.append({"stn": str(int(float(stn))), "name": r.get("STN_KO", ""), "lon": lon, "lat": lat})
    return out


def load_stations_csv(path):
    """지점 목록 CSV를 읽는다. 지점정보 API 대신 쓰는 대안.

    두 형식을 모두 받는다.
      1) 직접 만든 표: stn,name,lon,lat
      2) 기상자료개방포털 지점정보 내려받기 표: 지점, 지점명, 위도, 경도, (종료일) …
         같은 지점이 여러 줄(이력)이면 종료일이 비어 있는(지금 운영 중인) 줄을 쓴다.
    """
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return []

    def pick(r, *names):
        for n in names:
            for k in r:
                if k and k.strip().lower() == n:
                    return (r[k] or "").strip()
        return ""

    out = {}
    for r in rows:
        stn = pick(r, "stn", "지점", "지점번호", "stn_id")
        lat = pick(r, "lat", "위도")
        lon = pick(r, "lon", "경도")
        if not stn or not lat or not lon:
            continue
        try:
            item = {"stn": str(int(float(stn))), "name": pick(r, "name", "지점명"),
                    "lon": float(lon), "lat": float(lat)}
        except ValueError:
            continue
        ended = pick(r, "종료일")
        if item["stn"] not in out or not ended:
            out[item["stn"]] = item
    return list(out.values())


def fetch_minute(auth_key, tm=None):
    """전 지점 매분자료 → {stn: {...}}"""
    params = {"stn": "0", "disp": "1", "help": "1", "authKey": auth_key}
    if tm is not None:
        params["tm2"] = tm.strftime("%Y%m%d%H%M")
    text = _get_text(MIN_URL, params)
    rows = _parse_table(text, ["STN", "RN-15m"])
    out = {}
    for r in rows:
        try:
            stn = str(int(float(r["STN"])))
        except (KeyError, ValueError):
            continue
        out[stn] = {
            "time": r.get("YYMMDDHHMI"),
            "rn15": _num(r.get("RN-15m")),
            "rn60": _num(r.get("RN-60m")),
            "rnday": _num(r.get("RN-DAY")),
            "rain_now": r.get("RE") == "1",
            "ta": _num(r.get("TA")),
            "hm": _num(r.get("HM")),
            "ws": _num(r.get("WS10")),
            "wd": _num(r.get("WD10")),
        }
    return out


def kst_time(s):
    try:
        return dt.datetime.strptime(s, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None
