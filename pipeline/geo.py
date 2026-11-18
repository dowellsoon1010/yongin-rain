"""경계(GeoJSON) 읽기, 점-다각형 판정, 좌표 투영.

외부 GIS 라이브러리(shapely, pyproj) 없이 numpy만으로 동작하게 만들었다.
GitHub Actions 같은 곳에서 설치 부담을 줄이기 위해서다.
"""
import json
import math

import numpy as np


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------
def load_districts(path, name_field):
    """GeoJSON을 읽어 [{name, rings:[(N,2) array, ...]}] 형태로 돌려준다.

    rings 는 바깥 경계와 구멍을 모두 담는다. 짝-홀 규칙으로 판정하므로
    구멍이 있어도 맞게 계산된다. 좌표는 경도, 위도(EPSG:4326)여야 한다.
    """
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)

    out = []
    for feat in gj["features"]:
        props = feat.get("properties") or {}
        if name_field not in props:
            raise KeyError(
                f"경계 파일에 '{name_field}' 속성이 없습니다. 있는 속성: {list(props)}"
            )
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            polys = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        rings = [np.asarray(r, dtype=float)[:, :2] for poly in polys for r in poly]
        out.append({"name": str(props[name_field]), "rings": rings})

    # 좌표가 위경도인지 간단히 확인 (TM 좌표로 내보내면 값이 수십만 단위가 된다)
    sample = out[0]["rings"][0]
    if np.abs(sample).max() > 360:
        raise ValueError(
            "경계 좌표가 위경도가 아닙니다. QGIS에서 내보낼 때 좌표계를 EPSG:4326으로 지정하세요."
        )
    return out


def bbox_of(districts, margin=0.0):
    pts = np.vstack([r for d in districts for r in d["rings"]])
    return (
        pts[:, 0].min() - margin,
        pts[:, 1].min() - margin,
        pts[:, 0].max() + margin,
        pts[:, 1].max() + margin,
    )


def points_in_rings(x, y, rings):
    """짝-홀 규칙 점-다각형 판정. x, y 는 같은 모양의 배열."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    for ring in rings:
        xs, ys = ring[:, 0], ring[:, 1]
        x1, y1 = xs, ys
        x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
        for a, b, c, d in zip(x1, y1, x2, y2):
            if b == d:
                continue
            cond = (b > y) != (d > y)
            xint = a + (y - b) * (c - a) / (d - b)
            inside ^= cond & (x < xint)
    return inside


def which_district(lon, lat, districts):
    for d in districts:
        if points_in_rings(np.array([lon]), np.array([lat]), d["rings"])[0]:
            return d["name"]
    return None


def cell_area_m2(lats, step):
    """WGS84 타원체에서 위도별 격자 한 칸(step°×step°)의 면적(m²)."""
    a, e2 = 6378137.0, 0.00669437999014
    phi = np.radians(lats)
    w = np.sin(phi) ** 2
    dlon = math.pi / 180 * a * np.cos(phi) / np.sqrt(1 - e2 * w) * step
    dlat = math.pi / 180 * a * (1 - e2) / (1 - e2 * w) ** 1.5 * step
    return dlon * dlat


# ---------------------------------------------------------------------------
# 람베르트 정각원추도법(LCC) — 기상청 레이더 합성자료 격자
#   +proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +ellps=WGS84
#   (기상청 기상기후데이터위키의 레이더 예제와 같은 정의)
# ---------------------------------------------------------------------------
_A = 6378137.0
_E = 0.0818191908426215  # WGS84 이심률


def _m(phi):
    return np.cos(phi) / np.sqrt(1 - (_E * np.sin(phi)) ** 2)


def _t(phi):
    es = _E * np.sin(phi)
    return np.tan(math.pi / 4 - phi / 2) / ((1 - es) / (1 + es)) ** (_E / 2)


class LCC:
    def __init__(self, lat1=30.0, lat2=60.0, lat0=38.0, lon0=126.0):
        p1, p2, p0 = map(math.radians, (lat1, lat2, lat0))
        m1, m2 = _m(p1), _m(p2)
        t1, t2, t0 = _t(p1), _t(p2), _t(p0)
        self.n = (math.log(m1) - math.log(m2)) / (math.log(t1) - math.log(t2))
        self.F = m1 / (self.n * t1 ** self.n)
        self.rho0 = _A * self.F * t0 ** self.n
        self.lon0 = math.radians(lon0)

    def forward(self, lon, lat):
        """경도·위도(도) → LCC x, y(m)."""
        lam = np.radians(lon)
        phi = np.radians(lat)
        rho = _A * self.F * _t(phi) ** self.n
        theta = self.n * (lam - self.lon0)
        return rho * np.sin(theta), self.rho0 - rho * np.cos(theta)
