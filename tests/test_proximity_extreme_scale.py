"""极大/极小有限坐标下的近距离相遇复核与接触传播追踪测试。

距离平方在极大有限坐标下会溢出 float64（如相距 1e155 米时平方为 1e310 → inf），
曾导致相遇漏报、接触追踪误判不可达。相遇区间计算对相对位置、相对速度与阈值
按同一比例缩放后再解同一抛物线：有限输入的距离判定、区间端点与最小距离正确；
仅距离严格小于阈值的正时长计入，恰等阈值或瞬时接触不计，接点不满足条件不合并。

端点本身可取合法极大有限值 ±1e308：相邻轨迹点线性插值时 (pb-pa) 会溢出为
inf（pa + inf·r 把真实有限的中点算成 inf/NaN），GEOS 对 1e308 量级线段求交
也会在内部平方/叉积处溢出产生空覆盖。插值改走凸组合、几何求交按坐标量级
同比例缩放后，跨越正负范围的线性轨迹相遇区间、最小距离仍为有限值且正确。
"""
import math

import pytest

from tests.conftest import DAY_WINDOWS

T0 = "2026-09-30T10:00:00Z"


def _upload(client, trip, points):
    r = client.post(
        f"/api/v1/trips/{trip}/points",
        json={"points": [{"point_id": pid, "ts": ts, "x": x, "y": y} for pid, ts, x, y in points]},
    )
    assert r.status_code == 200, r.text


def _register(client, zone_id, exterior):
    r = client.post(
        "/api/v1/zones",
        json={"zone_id": zone_id, "exterior": exterior, "holes": [], "windows": DAY_WINDOWS},
    )
    assert r.status_code in (200, 201), r.text


def _proximity(client, zone_id, trip_ids, max_distance, since=T0, until="2026-09-30T10:05:00Z"):
    return client.post(
        f"/api/v1/zones/{zone_id}/proximity",
        json={"trip_ids": trip_ids, "since": since, "until": until, "max_distance": max_distance},
    )


def _tracing(client, zone_id, source, trip_ids, max_distance, since=T0, until="2026-09-30T10:05:00Z"):
    return client.post(
        f"/api/v1/zones/{zone_id}/contact-tracing",
        json={
            "source_trip_id": source,
            "trip_ids": trip_ids,
            "since": since,
            "until": until,
            "max_distance": max_distance,
        },
    )


def _epoch(iso: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


BIG_ZONE = [[-2e155, -2e155], [2e155, -2e155], [2e155, 2e155], [-2e155, 2e155]]


def test_stationary_huge_distance_encounter_reported_case(client):
    """漏报原案：两车在生效区内连续静止 10 秒、相距 1e155 米、阈值 2e155 米，
    距离平方溢出曾导致空相遇；修复后返回整段相遇与最小距离。"""
    _register(client, "big1", BIG_ZONE)
    _upload(client, "T1", [("a", T0, 1e155, 1e155), ("b", "2026-09-30T10:00:10Z", 1e155, 1e155)])
    _upload(client, "T2", [("a", T0, 2e155, 1e155), ("b", "2026-09-30T10:00:10Z", 2e155, 1e155)])

    r = _proximity(client, "big1", ["T1", "T2"], 2e155, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_distance"] == 2e155
    assert len(body["pairs"]) == 1
    seg = body["pairs"][0]["intervals"][0]
    assert seg["start"] == T0
    assert seg["end"] == "2026-09-30T10:00:10Z"
    assert seg["duration_sec"] == 10.0
    assert seg["min_distance"] == 1e155

    # 依赖相遇结果的接触传播追踪：T2 可达（1 跳，查询起点即到达）
    r = _tracing(client, "big1", "T1", ["T1", "T2"], 2e155, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["T1"]["reachable"] is True and results["T1"]["hops"] == 0
    assert results["T2"]["reachable"] is True
    assert results["T2"]["arrived_at"] == T0
    assert results["T2"]["hops"] == 1
    assert results["T2"]["chain"] == ["T1", "T2"]


def test_huge_distance_strict_threshold_boundary(client):
    """极大尺度下严格小于：距离恰等阈值不计入（接触追踪同样不可达）；
    阈值再大一格浮点（nextafter）则整段计入。"""
    _register(client, "big2", BIG_ZONE)
    _upload(client, "T1", [("a", T0, 0.0, 0.0), ("b", "2026-09-30T10:00:10Z", 0.0, 0.0)])
    _upload(client, "T2", [("a", T0, 1e155, 0.0), ("b", "2026-09-30T10:00:10Z", 1e155, 0.0)])

    # 恰等阈值：空相遇、不可达
    r = _proximity(client, "big2", ["T1", "T2"], 1e155, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200
    assert r.json()["pairs"] == []
    r = _tracing(client, "big2", "T1", ["T1", "T2"], 1e155, until="2026-09-30T10:00:10Z")
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["T2"]["reachable"] is False
    assert results["T2"]["arrived_at"] is None

    # 阈值大一格浮点：整段相遇
    r = _proximity(
        client, "big2", ["T1", "T2"], math.nextafter(1e155, math.inf),
        until="2026-09-30T10:00:10Z",
    )
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["duration_sec"] == 10.0
    assert seg["min_distance"] == 1e155


def test_huge_scale_moving_endpoints_and_min_distance(client):
    """大尺度相对运动：抛物线根（区间端点）与区间最小距离在缩放下保持正确。"""
    _register(client, "big3", BIG_ZONE)
    # A(t) = (-1e155 + 2e153·t, 0)，t∈[0,100]；B 静止 (0, 5e154)；阈值 1e155
    # |r(t)|² < 1e310 ⟺ |2e153·t - 1e155| < sqrt(7.5e309) → t ∈ (6.6987298…, 93.3012701…)
    _upload(client, "T1", [("a", T0, -1e155, 0.0), ("b", "2026-09-30T10:01:40Z", 1e155, 0.0)])
    _upload(client, "T2", [("a", T0, 0.0, 5e154), ("b", "2026-09-30T10:01:40Z", 0.0, 5e154)])
    r = _proximity(client, "big3", ["T1", "T2"], 1e155)
    assert r.status_code == 200, r.text
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    seg = segs[0]
    t0 = _epoch(T0)
    assert _epoch(seg["start"]) == pytest.approx(t0 + 6.698729810778065, abs=1e-6)
    assert _epoch(seg["end"]) == pytest.approx(t0 + 93.30127018922193, abs=1e-6)
    assert seg["duration_sec"] == 86.60254
    assert seg["min_distance"] == 5e154  # 顶点处恰为 5e154（缩放后精确表示）


def test_huge_scale_joint_equal_threshold_splits_and_merges(client):
    """极大尺度下的接点规则：接点距离恰等阈值时两段不合并；阈值大一格浮点则合并。"""
    _register(client, "big4", [[-1e154, -1e154], [2e155, -1e154], [2e155, 1e155], [-1e154, 1e155]])
    _upload(client, "J1", [
        ("a", T0, 0.0, 0.0),
        ("m", "2026-09-30T10:00:50Z", 5e154, 0.0),
        ("b", "2026-09-30T10:01:40Z", 1e155, 0.0),
    ])
    # 接点 t=50 处相距恰 2.5e154，两侧各线性拉开到 2.4e154
    _upload(client, "J2", [
        ("a", T0, 0.0, 2.4e154),
        ("m", "2026-09-30T10:00:50Z", 5e154, 2.5e154),
        ("b", "2026-09-30T10:01:40Z", 1e155, 2.4e154),
    ])
    r = _proximity(client, "big4", ["J1", "J2"], 2.5e154)
    segs = r.json()["pairs"][0]["intervals"]
    assert [(s["start"], s["end"], s["duration_sec"], s["min_distance"]) for s in segs] == [
        (T0, "2026-09-30T10:00:50Z", 50.0, 2.4e154),
        ("2026-09-30T10:00:50Z", "2026-09-30T10:01:40Z", 50.0, 2.4e154),
    ]
    # 阈值大一格浮点：接点严格满足，合并为一段
    r = _proximity(client, "big4", ["J1", "J2"], math.nextafter(2.5e154, math.inf))
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    assert segs[0]["start"] == T0
    segs[0]["end"] == "2026-09-30T10:01:40Z"
    assert segs[0]["duration_sec"] == 100.0
    assert segs[0]["min_distance"] == 2.4e154


def test_huge_threshold_with_normal_distance(client):
    """普通距离、极大阈值（阈值平方溢出）：整段相遇，最小距离仍为真实值。"""
    _register(client, "big5", [[0, 0], [100, 0], [100, 100], [0, 100]])
    _upload(client, "T1", [("a", T0, 0.0, 40.0), ("b", "2026-09-30T10:01:40Z", 100.0, 40.0)])
    _upload(client, "T2", [("a", T0, 0.0, 60.0), ("b", "2026-09-30T10:01:40Z", 100.0, 60.0)])
    r = _proximity(client, "big5", ["T1", "T2"], 1e155)
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == T0
    assert seg["end"] == "2026-09-30T10:01:40Z"
    assert seg["duration_sec"] == 100.0
    assert seg["min_distance"] == 20.0


def test_tiny_scale_encounter(client):
    """极小有限距离（平方下溢为 0 曾误判）：相距 1e-155、阈值 2e-155 整段相遇；
    阈值恰等 1e-155 不计。"""
    _register(client, "tiny1", [[-1, -1], [1, -1], [1, 1], [-1, 1]])
    _upload(client, "T1", [("a", T0, 0.0, 0.0), ("b", "2026-09-30T10:00:10Z", 0.0, 0.0)])
    _upload(client, "T2", [("a", T0, 1e-155, 0.0), ("b", "2026-09-30T10:00:10Z", 1e-155, 0.0)])
    r = _proximity(client, "tiny1", ["T1", "T2"], 2e-155, until="2026-09-30T10:00:10Z")
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["duration_sec"] == 10.0
    assert seg["min_distance"] == 1e-155
    r = _proximity(client, "tiny1", ["T1", "T2"], 1e-155, until="2026-09-30T10:00:10Z")
    assert r.json()["pairs"] == []


def test_opposite_sign_huge_coordinates_no_encounter(client):
    """异号极大坐标：真实距离 2e308 超出 float64 有限范围，必然不小于任何
    有限阈值（含最大有限浮点值）——合法无相遇返回空结果而非溢出误判。"""
    _register(client, "big6", [[-1.5e308, -1e308], [1.5e308, -1e308], [1.5e308, 1e308], [-1.5e308, 1e308]])
    _upload(client, "O1", [("a", T0, 1e308, 0.0), ("b", "2026-09-30T10:00:10Z", 1e308, 0.0)])
    _upload(client, "O2", [("a", T0, -1e308, 0.0), ("b", "2026-09-30T10:00:10Z", -1e308, 0.0)])
    r = _proximity(client, "big6", ["O1", "O2"], 1.7976931348623157e308, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    assert r.json()["pairs"] == []
    r = _tracing(client, "big6", "O1", ["O1", "O2"], 1.7976931348623157e308, until="2026-09-30T10:00:10Z")
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["O2"]["reachable"] is False


def test_huge_scale_instantaneous_crossing_ignored(client):
    """极大尺度下高速穿越阈值圆：正时长窗口短于零时长判定（<1e-9 秒），
    属瞬时接触，不计入相遇。"""
    _register(client, "big7", [[-1e308, -1e308], [1e308, -1e308], [1e308, 1e308], [-1e308, 1e308]])
    _upload(client, "X1", [("a", T0, -8e307, 0.0), ("b", "2026-09-30T10:00:10Z", 8e307, 0.0)])
    _upload(client, "X2", [("a", T0, 0.0, 1e100), ("b", "2026-09-30T10:00:10Z", 0.0, 1e100)])
    r = _proximity(client, "big7", ["X1", "X2"], 1e155, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    assert r.json()["pairs"] == []


def test_contact_tracing_huge_scale_multi_hop(client):
    """接触传播追踪在极大尺度下的多跳接续：T1→T2→T3 两跳可达；
    阈值收紧后源头不可达、全链不可达。"""
    _register(client, "big8", [[-1e155, -1e155], [4e155, -1e155], [4e155, 1e155], [-1e155, 1e155]])
    _upload(client, "T1", [("a", T0, 0.0, 0.0), ("b", "2026-09-30T10:00:10Z", 0.0, 0.0)])
    _upload(client, "T2", [("a", T0, 1e155, 0.0), ("b", "2026-09-30T10:00:10Z", 1e155, 0.0)])
    # T3 距 T1 2.9e155（> 阈值，不能直达）、距 T2 1.9e155（< 阈值，可接续）
    _upload(client, "T3", [("a", T0, 2.9e155, 0.0), ("b", "2026-09-30T10:00:10Z", 2.9e155, 0.0)])
    r = _tracing(client, "big8", "T1", ["T1", "T2", "T3"], 2e155, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["T1"]["hops"] == 0
    assert results["T2"]["reachable"] is True and results["T2"]["hops"] == 1
    # T3 只能经 T1→T2→T3 两跳到达
    assert results["T3"]["reachable"] is True
    assert results["T3"]["hops"] == 2
    assert results["T3"]["chain"] == ["T1", "T2", "T3"]
    # 阈值收紧到 7e154：T1↔T2（1e155）不可传递，全链不可达
    r = _tracing(client, "big8", "T1", ["T1", "T2", "T3"], 7e154, until="2026-09-30T10:00:10Z")
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["T2"]["reachable"] is False
    assert results["T3"]["reachable"] is False


def test_normal_scale_behavior_unchanged(client):
    """常规模值路径逐位保留：既有抛物线解析结果不变。"""
    _register(client, "keep1", [[0, 0], [100, 0], [100, 100], [0, 100]])
    _upload(client, "T1", [("a", T0, 0.0, 40.0), ("b", "2026-09-30T10:01:40Z", 100.0, 40.0)])
    _upload(client, "T2", [("a", T0, 0.0, 90.0), ("b", "2026-09-30T10:01:40Z", 100.0, 50.0)])
    r = _proximity(client, "keep1", ["T1", "T2"], 25)
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == "2026-09-30T10:01:02.500000Z"
    assert seg["end"] == "2026-09-30T10:01:40Z"
    assert seg["duration_sec"] == 37.5
    assert seg["min_distance"] == 10.0


def _assert_all_finite(obj):
    """递归断言响应中不出现 Infinity/NaN（保证 JSON 可序列化）。"""
    if isinstance(obj, float):
        assert math.isfinite(obj), obj
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_all_finite(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_all_finite(v)


def test_endpoints_pm_1e308_crossing_sign_range_encounter(client):
    """端点 -1e308 → 1e308 线性跨越正负范围：段内正时长相遇必须报出。

    相对运动分量（2e308 / 10 秒）与端点差均超出常规缩放路径，凸组合插值与
    缩放几何保证覆盖区间完整；阈值圆在段前段后截断，相遇区间端点与区间
    最小距离均为有限值。
    """
    _register(
        client, "span1",
        [[-1.5e308, -1.5e308], [1.5e308, -1.5e308], [1.5e308, 1.5e308], [-1.5e308, 1.5e308]],
    )
    # A: (-1e308, 0) → (1e308, 100)；B: (-1e308, 0) → (1e308, 0)，10 秒
    # 相对 y = 10t，严格小于阈值 50 ⟺ t ∈ [0, 5)
    _upload(client, "S1", [
        ("a", T0, -1e308, 0.0),
        ("b", "2026-09-30T10:00:10Z", 1e308, 100.0),
    ])
    _upload(client, "S2", [
        ("a", T0, -1e308, 0.0),
        ("b", "2026-09-30T10:00:10Z", 1e308, 0.0),
    ])
    r = _proximity(client, "span1", ["S1", "S2"], 50.0, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    _assert_all_finite(r.json())
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == T0
    assert seg["end"] == "2026-09-30T10:00:05Z"
    assert seg["duration_sec"] == 5.0
    assert seg["min_distance"] == 0.0

    # 严格小于：阈值恰为段末距离 50 时不计段末（区间仍在 [0,5)，正时长保留）
    # 接触传播沿用同一相遇区间：S2 在查询起点即可达
    r = _tracing(client, "span1", "S1", ["S1", "S2"], 50.0, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    _assert_all_finite(r.json())
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["S2"]["reachable"] is True
    assert results["S2"]["arrived_at"] == T0
    assert results["S2"]["hops"] == 1
    assert results["S2"]["chain"] == ["S1", "S2"]


def test_pm_1e308_segment_interior_cut_by_window_boundary(client):
    """生效窗口边界落在 ±1e308 线性段内部：段内凸组合插值给出有限位置，
    相遇区间从窗口半开起点开始（此前两车虽近但窗口未生效，不计）。"""
    r = client.post(
        "/api/v1/zones",
        json={
            "zone_id": "span2",
            "exterior": [[-1.5e308, -1.5e308], [1.5e308, -1.5e308],
                         [1.5e308, 1.5e308], [-1.5e308, 1.5e308]],
            "holes": [],
            "windows": [{"start": "2026-09-30T10:00:02Z", "end": "2026-09-30T10:00:11Z"}],
        },
    )
    assert r.status_code in (200, 201), r.text
    _upload(client, "W1", [
        ("a", T0, -1e308, 0.0),
        ("b", "2026-09-30T10:00:10Z", 1e308, 100.0),
    ])
    _upload(client, "W2", [
        ("a", T0, -1e308, 0.0),
        ("b", "2026-09-30T10:00:10Z", 1e308, 0.0),
    ])
    r = _proximity(client, "span2", ["W1", "W2"], 50.0, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    _assert_all_finite(r.json())
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == "2026-09-30T10:00:02Z"  # 半开窗口起点
    assert seg["end"] == "2026-09-30T10:00:05Z"
    assert seg["duration_sec"] == 3.0
    assert seg["min_distance"] == 20.0  # t=2 处相对 y=20

    r = _tracing(client, "span2", "W1", ["W1", "W2"], 50.0, until="2026-09-30T10:00:10Z")
    _assert_all_finite(r.json())
    results = {x["trip_id"]: x for x in r.json()["results"]}
    assert results["W2"]["reachable"] is True
    assert results["W2"]["arrived_at"] == "2026-09-30T10:00:02Z"


def test_pm_1e308_segments_joined_at_sample_strict_threshold_split(client):
    """±1e308 量级多段轨迹在采样接点处距离恰等阈值：两段不合并（区间合并
    语义不变）；阈值大一格浮点则合并为一段，全部数值有限。"""
    _register(
        client, "span3",
        [[-1.5e308, -1.5e308], [1.5e308, -1.5e308], [1.5e308, 1.5e308], [-1.5e308, 1.5e308]],
    )
    # 两车 x 轨迹完全相同（接点 t=5 位于中点）；y 偏移在接点恰为 1e155
    _upload(client, "Jx1", [
        ("a", T0, -1e308, 0.0),
        ("m", "2026-09-30T10:00:05Z", 0.0, 0.0),
        ("b", "2026-09-30T10:00:10Z", 1e308, 0.0),
    ])
    _upload(client, "Jx2", [
        ("a", T0, -1e308, 0.9e155),
        ("m", "2026-09-30T10:00:05Z", 0.0, 1e155),
        ("b", "2026-09-30T10:00:10Z", 1e308, 0.9e155),
    ])
    r = _proximity(client, "span3", ["Jx1", "Jx2"], 1e155, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    _assert_all_finite(r.json())
    segs = r.json()["pairs"][0]["intervals"]
    assert [(s["start"], s["end"], s["duration_sec"]) for s in segs] == [
        (T0, "2026-09-30T10:00:05Z", 5.0),
        ("2026-09-30T10:00:05Z", "2026-09-30T10:00:10Z", 5.0),
    ]
    r = _proximity(
        client, "span3", ["Jx1", "Jx2"], math.nextafter(1e155, math.inf),
        until="2026-09-30T10:00:10Z",
    )
    _assert_all_finite(r.json())
    segs = r.json()["pairs"][0]["intervals"]
    assert len(segs) == 1
    assert segs[0]["start"] == T0
    assert segs[0]["end"] == "2026-09-30T10:00:10Z"
    assert segs[0]["duration_sec"] == 10.0
    assert segs[0]["min_distance"] == 9e154


def test_pm_1e308_stationary_small_offset_encounter(client):
    """±1e308 端点上同向跨越、垂直偏移为常规小量（平方下溢路径）：整段相遇，
    最小距离为真实小值而非 0 或 inf/NaN。"""
    _register(
        client, "span4",
        [[-1.5e308, -1.5e308], [1.5e308, -1.5e308], [1.5e308, 1.5e308], [-1.5e308, 1.5e308]],
    )
    _upload(client, "P1", [
        ("a", T0, -1e308, 0.0),
        ("b", "2026-09-30T10:00:10Z", 1e308, 0.0),
    ])
    _upload(client, "P2", [
        ("a", T0, -1e308, 20.0),
        ("b", "2026-09-30T10:00:10Z", 1e308, 20.0),
    ])
    r = _proximity(client, "span4", ["P1", "P2"], 25.0, until="2026-09-30T10:00:10Z")
    assert r.status_code == 200, r.text
    # 原始 JSON 文本中也不得出现 Infinity/NaN（FastAPI 默认序列化会抛错或写出非法 JSON）
    assert "Infinity" not in r.text and "NaN" not in r.text
    seg = r.json()["pairs"][0]["intervals"][0]
    assert seg["start"] == T0
    assert seg["end"] == "2026-09-30T10:00:10Z"
    assert seg["duration_sec"] == 10.0
    assert seg["min_distance"] == 20.0
    # 阈值恰等 20：严格小于不成立
    r = _proximity(client, "span4", ["P1", "P2"], 20.0, until="2026-09-30T10:00:10Z")
    assert r.json()["pairs"] == []
