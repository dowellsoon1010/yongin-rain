"""기상청 API허브 연결 점검 (인증키는 화면에 표시하지 않음).

  set KMA_API_KEY=인증키
  py pipeline\\check_api.py

이 프로젝트가 쓰는 API를 하나씩 불러 보고, 성공 여부와 서버 안내문을 보여준다.
"""
import datetime as dt
import os
import sys

import requests

KEY = os.environ.get("KMA_API_KEY", "").strip()
if not KEY:
    sys.exit("KMA_API_KEY 가 비어 있습니다. 먼저 set KMA_API_KEY=인증키 를 입력하세요.")

now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).replace(tzinfo=None)
tm = now - dt.timedelta(minutes=now.minute % 5 + 15)
TM = tm.strftime("%Y%m%d%H%M")

TESTS = [
    ("레이더합성자료 다운로드 (HSR, 이진)", "https://apihub.kma.go.kr/api/typ04/url/rdr_cmp_file.php",
     {"tm": TM, "data": "bin", "cmp": "hsr"}, "binary"),
    ("AWS 매분자료", "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min",
     {"stn": "0", "disp": "1", "help": "1"}, "text"),
    ("지점정보 (AWS)", "https://apihub.kma.go.kr/api/typ01/url/stn_inf.php",
     {"inf": "AWS", "stn": "", "help": "1"}, "text"),
]


def readable(b):
    for enc in ("utf-8", "euc-kr"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", "replace")


print(f"인증키 길이 {len(KEY)}자 (앞 3자리 {KEY[:3]}…) / 시험 시각 {TM}\n")
for name, url, params, kind in TESTS:
    print("=" * 60)
    print(name)
    print("주소:", url)
    try:
        r = requests.get(url, params={**params, "authKey": KEY}, timeout=60)
    except Exception as e:  # noqa: BLE001
        print("  연결 실패:", str(e).replace(KEY, "***")[:300])
        continue
    c = r.content
    is_bin = c[:2] in (b"\x1f\x8b", b"PK")
    ok = r.status_code == 200 and (is_bin if kind == "binary" else b"#" in c[:2000])
    print(f"  결과: {'성공' if ok else '실패'} (HTTP {r.status_code}, {len(c):,} bytes)")
    if not ok or kind == "text":
        text = readable(c[:600]).strip()
        print("  서버 응답 앞부분:")
        for line in text.splitlines()[:12]:
            print("   ", line.replace(KEY, "***")[:150])
    if r.status_code == 403:
        print("  → 이 API에 대한 활용신청(승인)이 없거나, 신청한 API와 주소가 다를 수 있습니다.")
    elif r.status_code == 401:
        print("  → 인증키가 틀렸거나 만료되었습니다.")
print("=" * 60)
