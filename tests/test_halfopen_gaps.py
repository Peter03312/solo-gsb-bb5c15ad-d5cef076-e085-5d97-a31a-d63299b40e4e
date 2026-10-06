"""半开窗口真实交集回归测试：正时长空档（哪怕仅 1 微秒）必须切断占用与相遇。

背景：旧核算在时间域合并时使用 1e-6 秒容差，会把仅相隔 1 微秒的两个 UTC
生效窗口视为连续——窗口外时间被计入穿越时长，两车相遇区间也可能跨越空档，
使接触传播追踪提前到达。本测试固定下列语义：

- 区域窗口、轨迹覆盖与查询范围按**真实半开交集**核算；
- **正时长空档必须切断**占用（穿越/同时占用/风险预算/超速/许可/工序）与相遇；
- 仅**端点相接且条件连续**时合并；
- 距离严格小于阈值、仅正时长相遇可传递；无可传递相遇时标记 reachable=false。
"""
from tests.conftest import SQUARE, register_square

BASE = "2026-10-05T12:00:00Z"

# 两个仅相隔 1 微秒的生效窗口：[12:00:00, 12:00:10) 与 [12:00:10.000001, 12:00:20)
WINDOWS_1US_GAP = [
    {"start": "2026-10-05T12:00:00Z", "end": "2026-10-05T12:00:10Z"},
    {"start": "2026-10-05T12:00:10.000001Z", "end": "2026-10-05T12:00:20Z"},
]
# 端点恰好相接：[12:00:00, 12:00:10) 与 [12:00:10, 12:00:20)，须合并为一段
WINDOWS_TOUCHING = [
    {"start": "2026-10-05T12:00:00Z", "end": "2026-10-05T12:00:10Z"},
    {"start": "2026-10-05T12:00:10Z", "end": "2026-10-05T12:00:20Z"},
]
# 中间有 2 秒正空档
WINDOWS_2S_GAP = [
    {"start": "2026-10-05T12:00:00Z", "end": "2026-10-05T12:00:09Z"},
    {"start": "2026-10-05T12:00:11Z", "end": "2026-10-05T12:00:20Z"},
]


def _upload(client, trip, points):
    r = client.post(
        f"/api/v1/trips/{trip}/points",
        json={"points": [{"point_id": pid, "ts": ts, "x": x, "y": y} for pid, ts, x, y in points]},
    )
    assert r.status_code == 200, r.text
    return r


# 全程位于区内、匀速从 (50,0) 到 (50,100) 的 20 秒轨迹
TRACK_T = [
    ("p1", "2026-10-05T12:00:00Z", 50, 0),
    ("p2", "2026-10-05T12:00:20Z", 50, 100),
]
# 与 T 平行、恒定相距 1m
TRACK_T2 = [
    ("q1", "2026-10-05T12:00:00Z", 51, 0),
    ("q2", "2026-10-05T12:00:20Z", 51, 100),
]


# ---------------------------------------------------------------------
# 穿越审计
# ---------------------------------------------------------------------

def test_crossing_one_microsecond_gap_splits_segments(client):
    """1 微秒空档把穿越切成两段，空档不计入累计时长。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    _upload(client, "T", TRACK_T)

    r = client.get("/api/v1/trips/T/crossings", params={"zone_id": "Z"})
    assert r.status_code == 200, r.text
    zone = r.json()["zones"][0]
    assert zone["crossed"] is True
    segments = zone["segments"]
    assert len(segments) == 2
    assert segments[0]["start"] == "2026-10-05T12:00:00Z"
    assert segments[0]["end"] == "2026-10-05T12:00:10Z"
    assert segments[0]["duration_sec"] == 10.0
    assert segments[1]["start"] == "2026-10-05T12:00:10.000001Z"
    assert segments[1]["end"] == "2026-10-05T12:00:20Z"
    assert segments[1]["duration_sec"] == 9.999999
    # 累计恰为两窗口并集长度（19.999999），不把 1 微秒空档计入
    assert zone["total_duration_sec"] == 19.999999


def test_crossing_touching_windows_remain_one_segment(client):
    """端点恰好相接（左闭右开）的窗口仍合并为单段 20 秒。"""
    register_square(client, zone_id="Z", windows=WINDOWS_TOUCHING)
    _upload(client, "T", TRACK_T)

    r = client.get("/api/v1/trips/T/crossings", params={"zone_id": "Z"})
    assert r.status_code == 200, r.text
    zone = r.json()["zones"][0]
    assert len(zone["segments"]) == 1
    assert zone["segments"][0] == {
        "start": "2026-10-05T12:00:00Z",
        "end": "2026-10-05T12:00:20Z",
        "duration_sec": 20.0,
    }
    assert zone["total_duration_sec"] == 20.0


def test_crossing_query_range_half_open_intersection(client):
    """查询范围与窗口按半开求交：查询止于空档内时只取第一段。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    _upload(client, "T", TRACK_T)

    r = client.get(
        "/api/v1/trips/T/crossings",
        params={
            "zone_id": "Z",
            "since": "2026-10-05T12:00:00Z",
            "until": "2026-10-05T12:00:10.0000005Z",
        },
    )
    assert r.status_code == 200, r.text
    zone = r.json()["zones"][0]
    assert len(zone["segments"]) == 1
    assert zone["segments"][0]["end"] == "2026-10-05T12:00:10Z"
    assert zone["total_duration_sec"] == 10.0


# ---------------------------------------------------------------------
# 多车同时占用
# ---------------------------------------------------------------------

def test_co_occupancy_gap_splits_intervals(client):
    """1 微秒空档期间区域不生效：同时占用区间被切成两段。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    _upload(client, "T", TRACK_T)
    _upload(client, "T2", TRACK_T2)

    r = client.post(
        "/api/v1/zones/Z/co-occupancy",
        json={
            "trip_ids": ["T", "T2"],
            "since": BASE,
            "until": "2026-10-05T12:00:20Z",
            "threshold": 2,
        },
    )
    assert r.status_code == 200, r.text
    intervals = r.json()["intervals"]
    assert len(intervals) == 2
    assert intervals[0]["start"] == "2026-10-05T12:00:00Z"
    assert intervals[0]["end"] == "2026-10-05T12:00:10Z"
    assert intervals[1]["start"] == "2026-10-05T12:00:10.000001Z"
    assert intervals[1]["end"] == "2026-10-05T12:00:20Z"
    assert all(iv["count"] == 2 and iv["trip_ids"] == ["T", "T2"] for iv in intervals)


# ---------------------------------------------------------------------
# 近距离相遇
# ---------------------------------------------------------------------

def test_proximity_gap_splits_encounter(client):
    """两车全程相距 1m，但 1 微秒空档内区域不生效：相遇区间必须分成两段。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    _upload(client, "T", TRACK_T)
    _upload(client, "T2", TRACK_T2)

    r = client.post(
        "/api/v1/zones/Z/proximity",
        json={
            "trip_ids": ["T", "T2"],
            "since": BASE,
            "until": "2026-10-05T12:00:20Z",
            "max_distance": 5,
        },
    )
    assert r.status_code == 200, r.text
    pairs = r.json()["pairs"]
    assert len(pairs) == 1
    intervals = pairs[0]["intervals"]
    assert len(intervals) == 2
    assert intervals[0]["start"] == "2026-10-05T12:00:00Z"
    assert intervals[0]["end"] == "2026-10-05T12:00:10Z"
    assert intervals[1]["start"] == "2026-10-05T12:00:10.000001Z"
    assert intervals[1]["end"] == "2026-10-05T12:00:20Z"
    assert round(sum(iv["duration_sec"] for iv in intervals), 6) == 19.999999
    assert all(iv["min_distance"] == 1.0 for iv in intervals)


def test_proximity_touching_windows_merged(client):
    """端点相接窗口下，连续满足距离条件的相遇合并为一段。"""
    register_square(client, zone_id="Z", windows=WINDOWS_TOUCHING)
    _upload(client, "T", TRACK_T)
    _upload(client, "T2", TRACK_T2)

    r = client.post(
        "/api/v1/zones/Z/proximity",
        json={
            "trip_ids": ["T", "T2"],
            "since": BASE,
            "until": "2026-10-05T12:00:20Z",
            "max_distance": 5,
        },
    )
    assert r.status_code == 200, r.text
    intervals = r.json()["pairs"][0]["intervals"]
    assert len(intervals) == 1
    assert intervals[0]["start"] == "2026-10-05T12:00:00Z"
    assert intervals[0]["end"] == "2026-10-05T12:00:20Z"
    assert intervals[0]["duration_sec"] == 20.0


# ---------------------------------------------------------------------
# 接触传播追踪
# ---------------------------------------------------------------------

def test_contact_tracing_gap_does_not_advance_arrival(client):
    """源车在 1 微秒空档内到达：第一段相遇已结束，下游最早只能在第二段起点到达。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    _upload(client, "T", TRACK_T)
    _upload(client, "T2", TRACK_T2)

    r = client.post(
        "/api/v1/zones/Z/contact-tracing",
        json={
            "source_trip_id": "T",
            "trip_ids": ["T", "T2"],
            "since": "2026-10-05T12:00:10.000000Z",
            "until": "2026-10-05T12:00:20Z",
            "max_distance": 5,
        },
    )
    assert r.status_code == 200, r.text
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["T"]["arrived_at"] == "2026-10-05T12:00:10Z"
    # 旧实现把两段相遇桥接成 [00,20)，会错误地在查询起点即到达 T2
    assert results["T2"]["reachable"] is True
    assert results["T2"]["arrived_at"] == "2026-10-05T12:00:10.000001Z"
    assert results["T2"]["hops"] == 1
    assert results["T2"]["chain"] == ["T", "T2"]


def test_contact_tracing_only_gap_query_marks_unreachable(client):
    """查询范围只落在正时长空档（2 秒）内：没有可传递相遇，下游明确不可达。"""
    register_square(client, zone_id="Z2", windows=WINDOWS_2S_GAP)
    _upload(client, "T", TRACK_T)
    _upload(client, "T2", TRACK_T2)

    r = client.post(
        "/api/v1/zones/Z2/contact-tracing",
        json={
            "source_trip_id": "T",
            "trip_ids": ["T", "T2"],
            "since": "2026-10-05T12:00:09.5Z",
            "until": "2026-10-05T12:00:10.5Z",
            "max_distance": 500,
        },
    )
    assert r.status_code == 200, r.text
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["T"]["reachable"] is True  # 源车在查询起点即到达
    assert results["T2"] == {
        "trip_id": "T2",
        "reachable": False,
        "arrived_at": None,
        "hops": None,
        "chain": None,
    }


def test_contact_tracing_chain_across_gap_not_early(client):
    """三车链 S→B→C，源车在空档内到达：旧桥接会把两车都标记为早 1 微秒。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    _upload(client, "S", TRACK_T)
    _upload(client, "B", TRACK_T2)
    # C 与 B 恒定相距 4.5m、与 S 相距 5.5m：只能经 S→B→C 两跳到达
    _upload(
        client,
        "C",
        [
            ("c1", "2026-10-05T12:00:00Z", 55.5, 0),
            ("c2", "2026-10-05T12:00:20Z", 55.5, 100),
        ],
    )

    r = client.post(
        "/api/v1/zones/Z/contact-tracing",
        json={
            "source_trip_id": "S",
            "trip_ids": ["S", "B", "C"],
            "since": "2026-10-05T12:00:10Z",
            "until": "2026-10-05T12:00:20Z",
            "max_distance": 5,
        },
    )
    assert r.status_code == 200, r.text
    results = {x["trip_id"]: x for x in r.json()["results"]}
    # 第一段相遇 [00,10) 在源车到达时已经结束（恰在终点也算结束），
    # B 与 C 最早都只能在第二段相遇起点 10.000001 到达——不得提前到空档内
    assert results["B"]["arrived_at"] == "2026-10-05T12:00:10.000001Z"
    assert results["B"]["chain"] == ["S", "B"]
    assert results["C"]["arrived_at"] == "2026-10-05T12:00:10.000001Z"
    assert results["C"]["hops"] == 2
    assert results["C"]["chain"] == ["S", "B", "C"]


# ---------------------------------------------------------------------
# 风险预算 / 超速 / 许可 / 工序（同一 _coverage_intervals 路径）
# ---------------------------------------------------------------------

def test_risk_budget_excludes_gap(client):
    """风险预算的加权秒数不含 1 微秒空档。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    _upload(client, "T", TRACK_T)

    r = client.post(
        "/api/v1/trips/T/risk-budget",
        json={
            "since": BASE,
            "until": "2026-10-05T12:00:20Z",
            "zone_weights": {"Z": 2.0},
            "budget_weighted_sec": 1000.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_weighted_sec"] == round(19.999999 * 2.0, 6)
    assert len(body["segments"]) == 2
    assert body["segments"][0]["duration_sec"] == 10.0
    assert body["segments"][1]["duration_sec"] == 9.999999


def test_speeding_review_excludes_gap(client):
    """超速核算在 1 微秒空档处分段，违规总时长不含空档。"""
    register_square(client, zone_id="Z", windows=WINDOWS_1US_GAP)
    # 5 m/s，限速 1 m/s，采样间隔 20s 不超限
    _upload(client, "T", TRACK_T)

    r = client.post(
        "/api/v1/trips/T/speeding-review",
        json={
            "since": BASE,
            "until": "2026-10-05T12:00:20Z",
            "zone_speed_limits": {"Z": 1.0},
            "max_sample_interval_sec": 60.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["segments"]) == 2
    assert body["total_violation_duration_sec"] == 19.999999
    assert body["segments"][0]["end"] == "2026-10-05T12:00:10Z"
    assert body["segments"][1]["start"] == "2026-10-05T12:00:10.000001Z"
