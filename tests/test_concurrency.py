"""并发测试：并发登记的修订号全局唯一且连续；查询读不到半次提交；
审计核算在离开读视图后进行，不独占数据库访问。"""
import threading

from tests.conftest import SQUARE


def test_concurrent_registrations_get_unique_monotonic_revisions():
    from app.main import db, service

    n_per_zone = 20
    errors = []

    def register(zone_id: str, start: int) -> None:
        try:
            for i in range(start, start + n_per_zone):
                # 同 ID 反复覆盖，逐步扩大外边界，保证每次内容合法且不同
                size = 100 + i
                service.register_zone(
                    {
                        "zone_id": zone_id,
                        "exterior": [[0, 0], [size, 0], [size, size], [0, size]],
                        "holes": [],
                        "windows": [
                            {"start": "2026-09-30T00:00:00Z", "end": "2026-10-01T00:00:00Z"}
                        ],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=register, args=("zone-x", 0)),
        threading.Thread(target=register, args=("zone-y", 1000)),
        threading.Thread(target=register, args=("zone-z", 2000)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    head = service.current_revision()
    assert head == 3 * n_per_zone

    # 修订号连续无缺号；每个修订号恰好一条不可变记录
    with db.read_view() as conn:
        rows = conn.execute(
            "SELECT revision FROM zone_revisions ORDER BY revision"
        ).fetchall()
    revisions = [r["revision"] for r in rows]
    assert revisions == list(range(1, head + 1))


def test_concurrent_queries_never_see_half_commit(client):
    # 先登记一个基础区并上传轨迹
    r = client.post(
        "/api/v1/zones",
        json={
            "zone_id": "live",
            "exterior": SQUARE,
            "holes": [],
            "windows": [{"start": "2026-09-30T00:00:00Z", "end": "2026-10-02T00:00:00Z"}],
        },
    )
    assert r.status_code == 201
    client.post(
        "/api/v1/trips/live-trip/points",
        json={
            "points": [
                {"point_id": "a", "ts": "2026-09-30T10:00:00Z", "x": -50, "y": 50},
                {"point_id": "b", "ts": "2026-09-30T10:01:20Z", "x": 150, "y": 50},
            ]
        },
    )

    stop = threading.Event()
    failures = []

    def writer() -> None:
        # 反复同 ID 覆盖更新（窗口/几何交替变化）
        i = 0
        while not stop.is_set():
            size = 100 + (i % 5) * 20
            client.post(
                "/api/v1/zones",
                json={
                    "zone_id": "live",
                    "exterior": [[0, 0], [size, 0], [size, size], [0, size]],
                    "holes": [],
                    "windows": [
                        {"start": "2026-09-30T00:00:00Z", "end": "2026-10-02T00:00:00Z"}
                    ],
                },
            )
            i += 1

    def reader() -> None:
        import random

        while not stop.is_set():
            # 最新读：必须始终返回结构完整的结果
            r = client.get("/api/v1/trips/live-trip/crossings?zone_id=live")
            if r.status_code != 200:
                failures.append(("latest", r.status_code, r.text))
                return
            zones = r.json().get("zones")
            if not zones or "total_duration_sec" not in zones[0]:
                failures.append(("latest-payload", r.text))
                return
            # 指定修订号读：任何已存在的修订号都必须可读、结果可复现
            head = client.get("/api/v1/revisions").json()["current_revision"]
            rev = random.randint(1, head)
            r1 = client.get(f"/api/v1/trips/live-trip/crossings?zone_id=live&revision={rev}")
            r2 = client.get(f"/api/v1/trips/live-trip/crossings?zone_id=live&revision={rev}")
            if r1.status_code != 200 or r2.status_code != 200:
                failures.append(("pinned-status", rev, r1.status_code, r2.status_code))
                return
            if r1.json() != r2.json():
                failures.append(("pinned-nondeterministic", rev))
                return

    threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    # 让并发混合跑约 1 秒
    stop.wait(timeout=1.0)
    stop.set()
    for t in threads:
        t.join(timeout=10)

    assert not failures


def test_audits_compute_outside_read_view_and_writes_not_blocked(client, monkeypatch):
    """两个审计请求在离开读视图后并行核算；核算期间区域登记不被阻塞。"""
    from app.main import service

    # 基础数据：全天危险区 Z + 两辆横穿的车
    assert client.post(
        "/api/v1/zones",
        json={
            "zone_id": "Z",
            "exterior": SQUARE,
            "holes": [],
            "windows": [{"start": "2026-09-30T00:00:00Z", "end": "2026-10-02T00:00:00Z"}],
        },
    ).status_code == 201
    for tid, pts in (
        ("t1", [("a", "2026-09-30T12:00:00Z", -20, 50), ("b", "2026-09-30T12:02:00Z", 120, 50)]),
        ("t2", [("a", "2026-09-30T12:00:30Z", 50, -20), ("b", "2026-09-30T12:02:30Z", 50, 120)]),
    ):
        assert client.post(
            f"/api/v1/trips/{tid}/points",
            json={"points": [
                {"point_id": pid, "ts": ts, "x": x, "y": y} for pid, ts, x, y in pts
            ]},
        ).status_code == 200

    # 把核算拖慢：进入 _coverage_intervals 即视为“已离开读视图、正在核算”
    original = type(service)._coverage_intervals
    entered_lock = threading.Lock()
    entered = []
    both_entered = threading.Event()
    release = threading.Event()

    def slow_coverage(self, zrow, points, lo, hi):
        with entered_lock:
            entered.append(1)
            if len(entered) >= 2:
                both_entered.set()
        release.wait(timeout=15)
        return original(self, zrow, points, lo, hi)

    monkeypatch.setattr(type(service), "_coverage_intervals", slow_coverage)

    body = {
        "trip_ids": ["t1", "t2"],
        "since": "2026-09-30T12:00:00Z",
        "until": "2026-09-30T12:05:00Z",
        "threshold": 2,
    }
    results = {}

    def audit(name):
        results[name] = client.post("/api/v1/zones/Z/co-occupancy", json=body)

    threads = [threading.Thread(target=audit, args=(n,)) for n in ("a", "b")]
    try:
        for t in threads:
            t.start()
        # 两个审计请求都已取数并同时进入核算：读视图均已释放，互不独占
        assert both_entered.wait(timeout=10)

        # 核算仍在进行，区域登记（写事务）不排队、立即完成
        write_status = []

        def write():
            r = client.post(
                "/api/v1/zones",
                json={
                    "zone_id": "Z-other",
                    "exterior": SQUARE,
                    "holes": [],
                    "windows": [
                        {"start": "2026-09-30T00:00:00Z", "end": "2026-10-02T00:00:00Z"}
                    ],
                },
            )
            write_status.append(r.status_code)

        wt = threading.Thread(target=write)
        wt.start()
        wt.join(timeout=5)
        assert write_status == [201]
    finally:
        release.set()
        for t in threads:
            t.join(timeout=20)

    assert results["a"].status_code == 200, results["a"].text
    assert results["b"].status_code == 200, results["b"].text
    for r in results.values():
        payload = r.json()
        # 两车同时在区的时段：12:00:47.142857 ~ 12:01:42.857143
        assert len(payload["intervals"]) == 1
        interval = payload["intervals"][0]
        assert interval["trip_ids"] == ["t1", "t2"]
        assert interval["count"] == 2
        assert abs(interval["duration_sec"] - 55.714286) < 1e-4


def test_read_view_releases_connection_and_transaction_on_error():
    """读视图建立/查询失败后连接与事务都释放：连接归还池中并被后续读视图复用。"""
    import sqlite3

    import pytest

    from app.main import db, service
    from app.service import NotFoundError

    with db.read_view() as conn1:
        conn1.execute("SELECT 1")
    assert not conn1.in_transaction

    # 视图内查询失败：事务回滚、连接归还
    with pytest.raises(sqlite3.Error):
        with db.read_view() as conn2:
            conn2.execute("SELECT * FROM no_such_table")
    assert not conn2.in_transaction

    # 服务层在读视图内主动报错（404）同样释放
    with pytest.raises(NotFoundError):
        service.get_zone("ghost-zone", 1)

    with db.read_view() as conn3:
        pass
    # 失败路径未泄漏连接：同一连接被复用
    assert conn1 is conn2 is conn3
