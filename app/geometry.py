"""平面米制多边形校验与线段-区域裁剪。

约定：
- 坐标为平面米制坐标，单位统一为米，不做投影换算。
- 外边界（含其边线）属于危险区；洞及其边界不属于危险区。
- 线段与区域的交集中、落在洞边界上的零宽部分需要剔除。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from shapely.geometry import LineString, Point, Polygon, LinearRing
from shapely.geometry.base import BaseGeometry
from shapely import is_valid_reason

Coord = Tuple[float, float]

# 数值容差：参数区间端点小于该值的差异视为相同
_PARAM_EPS = 1e-9


class GeometryError(ValueError):
    """几何数据非法（自交、洞越界、坐标非有限等）。"""


def finite_coord(x: object, y: object) -> Coord:
    """校验坐标分量均为有限实数，拒绝 NaN / Inf 及非数值。"""
    if isinstance(x, bool) or isinstance(y, bool):
        raise GeometryError("坐标分量必须是数值，不能为布尔值")
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"坐标分量无法解析为数值: ({x!r}, {y!r})") from exc
    if not (math.isfinite(xf) and math.isfinite(yf)):
        raise GeometryError(f"坐标必须为有限数值，禁止 NaN/Infinity: ({x!r}, {y!r})")
    return xf, yf


def _normalize_ring(ring: Sequence[Sequence[float]], name: str) -> List[Coord]:
    """校验并归一化一个环：返回闭合坐标列表（首尾重复）。"""
    if not isinstance(ring, (list, tuple)) or len(ring) < 3:
        raise GeometryError(f"{name}至少需要 3 个点，实际为 {len(ring) if isinstance(ring, (list, tuple)) else '非列表'}")
    pts: List[Coord] = []
    for raw in ring:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise GeometryError(f"{name}的每个顶点必须是 [x, y] 二元组")
        pts.append(finite_coord(raw[0], raw[1]))
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    if len(pts) < 4:
        raise GeometryError(f"{name}至少需要 3 个互不重复的有效顶点")
    # 重复（非首尾闭合）点会导致退化，明确拒绝
    body = pts[:-1]
    if len(set(body)) != len(body):
        raise GeometryError(f"{name}存在重复顶点（非闭合位置），环不合法")
    return pts


def _validation_scale(points: Sequence[Coord]) -> float:
    """合法但极大（>1e100 模长）的坐标在 GEOS 求面积/叉积时会溢出为
    inf/NaN；校验时同比例缩放到 O(1)（相似变换不改变合法性与拓扑）。"""
    magnitude = max((max(abs(x), abs(y)) for x, y in points), default=0.0)
    return magnitude if magnitude > _GEOS_SAFE_SCALE else 1.0


def build_polygon(
    exterior: Sequence[Sequence[float]],
    holes: Sequence[Sequence[Sequence[float]]] | None = None,
) -> Polygon:
    """构造并严格校验带洞多边形。

    拒绝：自交/自切的外环或洞环、洞越出外环、洞之间重叠。
    """
    ext_pts = _normalize_ring(exterior, "外边界")

    hole_pts: List[List[Coord]] = []
    hole_geoms: List[LinearRing] = []
    for index, hole in enumerate(holes or []):
        label = f"洞#{index + 1}"
        hpts = _normalize_ring(hole, label)
        hole_pts.append(hpts)

    # 校验几何（含面积）用同比例缩放副本：极大合法有限坐标（±1e308 量级）
    # 在 GEOS 内部平方/叉积会溢出 inf/NaN，相似缩放下合法性/包含关系不变
    scale = _validation_scale(
        ext_pts + [pt for hpts in hole_pts for pt in hpts]
    )
    ext_check = [(x / scale, y / scale) for x, y in ext_pts]
    holes_check = [[(x / scale, y / scale) for x, y in hpts] for hpts in hole_pts]

    for index, hpts in enumerate(holes_check):
        label = f"洞#{index + 1}"
        ring = LinearRing(hpts)
        if not ring.is_valid:
            raise GeometryError(f"{label}自交或退化，不是合法的简单环: {is_valid_reason(ring)}")
        hole_geoms.append(ring)

    ext_ring = LinearRing(ext_check)
    if not ext_ring.is_valid:
        raise GeometryError(f"外边界自交或退化，不是合法的简单环: {is_valid_reason(ext_ring)}")

    polygon_check = Polygon(ext_check, holes_check)

    # Polygon 构造会静默修复部分非法输入（如洞在外部），因此显式检查
    if not polygon_check.is_valid:
        raise GeometryError(f"多边形不合法: {is_valid_reason(polygon_check)}")
    if polygon_check.is_empty or polygon_check.area <= 0:
        raise GeometryError("多边形面积必须为正")

    outer = Polygon(ext_check)
    for index, hpts in enumerate(holes_check):
        label = f"洞#{index + 1}"
        hole_poly = Polygon(hpts)
        if hole_poly.area <= 0:
            raise GeometryError(f"{label}面积必须为正")
        # 洞必须严格位于外边界内部（不得越界，也不得与外边界共边/接触）
        if not outer.contains(hole_poly):
            if not outer.covers(hole_poly):
                raise GeometryError(f"{label}越出外边界范围")
            raise GeometryError(f"{label}与外边界相接，洞必须严格位于外边界内部")

    # 洞之间不得重叠或边界相交
    for i in range(len(hole_geoms)):
        for j in range(i + 1, len(hole_geoms)):
            if hole_geoms[i].intersects(hole_geoms[j]):
                raise GeometryError(f"洞#{i + 1} 与 洞#{j + 1} 相交或重叠，洞之间必须互不相交")

    # 返回原始坐标构造的多边形（存储/查询用），再做一次内部一致性检查
    polygon = Polygon(ext_pts, hole_pts)
    if not polygon.is_valid:
        raise GeometryError(f"多边形不合法: {is_valid_reason(polygon)}")
    return polygon


# ---------------------------------------------------------------------------
# 区间工具（参数 t ∈ [0,1]，随后映射到时间）
# ---------------------------------------------------------------------------

def _subtract_intervals(
    base: Sequence[Tuple[float, float]],
    cuts: Sequence[Tuple[float, float]],
    eps: float = _PARAM_EPS,
) -> List[Tuple[float, float]]:
    """从若干互不相交区间 base 中减去 cuts。"""
    result: List[Tuple[float, float]] = []
    for a, b in base:
        pieces = [(a, b)]
        for c, d in cuts:
            next_pieces: List[Tuple[float, float]] = []
            for pa, pb in pieces:
                if d <= pa + eps or c >= pb - eps:
                    next_pieces.append((pa, pb))
                    continue
                if c > pa:
                    next_pieces.append((pa, min(pb, c)))
                if d < pb:
                    next_pieces.append((max(pa, d), pb))
            pieces = [(p, q) for p, q in next_pieces if q - p > eps]
        result.extend(pieces)
    return result


def merge_intervals(
    intervals: Sequence[Tuple[float, float]],
    eps: float = _PARAM_EPS,
) -> List[Tuple[float, float]]:
    """合并重叠或端点相接的区间。"""
    ordered = sorted(intervals)
    merged: List[Tuple[float, float]] = []
    for a, b in ordered:
        if b - a <= eps:
            continue
        if not merged or a > merged[-1][1] + eps:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    return merged


def _line_pieces_on_geometry(
    segment: LineString, geom: BaseGeometry
) -> List[LineString]:
    """提取 LineString 与某几何相交的所有一维部分（忽略点接触）。"""
    inter = segment.intersection(geom)
    if inter.is_empty:
        return []
    pieces: List[LineString] = []
    if inter.geom_type == "LineString":
        candidates = [inter]
    elif inter.geom_type in ("MultiLineString", "GeometryCollection"):
        candidates = [g for g in inter.geoms if g.geom_type == "LineString"]
    else:
        candidates = []
    for piece in candidates:
        if piece.length > 0:
            pieces.append(piece)
    return pieces


def _params_of_piece(piece: LineString, p0: Point, p1: Point, vx: float, vy: float) -> Tuple[float, float]:
    """把交线子段的端点映射回参数 t ∈ [0,1]。"""
    coords = list(piece.coords)
    a_pt, b_pt = coords[0], coords[-1]

    def param_of(pt: Sequence[float]) -> float:
        if abs(vx) >= abs(vy):
            t = (pt[0] - p0.x) / vx if vx != 0 else 0.0
        else:
            t = (pt[1] - p0.y) / vy if vy != 0 else 0.0
        return min(1.0, max(0.0, t))

    ta, tb = param_of(a_pt), param_of(b_pt)
    if ta > tb:
        ta, tb = tb, ta
    return ta, tb


# GEOS（float64）内部会对坐标做平方/叉积：分量模长超过 ~1e154 时中间量
# 溢出为 inf/NaN（即使输入均为合法有限值，如端点 -1e308→1e308 的线段）。
# 超过该模长时把多边形与线段同比例缩放到 O(1) 再求交，拓扑/参数不变。
_GEOS_SAFE_SCALE = 1e100


def _scaled_polygon_and_segment(
    polygon: Polygon, p0: Coord, p1: Coord
) -> Tuple[Polygon, Coord, Coord, float]:
    """极大坐标模长时，把多边形（含洞环）与线段同除一个正比例。

    缩放是相似变换：线段与区域的相交拓扑不变，交点在线段上的参数 t∈[0,1]
    也不变（``_params_of_piece`` 只做比值），因此调用方得到的参数区间与
    不缩放时数学上一致，但 GEOS 内部不再溢出/产生 NaN。

    缩放下塌缩为重复点的洞环剔除：其真实尺寸小到任何正时长经过都无法
    在 float64 时间上体现，丢弃不影响可见语义。
    """
    rings = [list(polygon.exterior.coords)]
    rings.extend(list(ring.coords) for ring in polygon.interiors)
    magnitude = max(
        (abs(x) for ring in rings for x, _y in ring),
        default=0.0,
    )
    magnitude = max(magnitude, abs(p0[0]), abs(p0[1]), abs(p1[0]), abs(p1[1]))
    if magnitude <= _GEOS_SAFE_SCALE:
        return polygon, p0, p1, 1.0
    scale = magnitude

    def _scaled_ring(ring: Sequence[Coord]) -> Optional[List[Coord]]:
        coords = [(x / scale, y / scale) for x, y in ring]
        return coords if len(set(coords)) >= 4 else None

    ext = _scaled_ring(list(polygon.exterior.coords))
    sp0, sp1 = (p0[0] / scale, p0[1] / scale), (p1[0] / scale, p1[1] / scale)
    if ext is None or not all(math.isfinite(v) for pt in (sp0, sp1) for v in pt):
        # 外环都缩没了或线段仍非有限：任何正时长子区间在 float64 下都不可表示
        return None, sp0, sp1, scale
    holes: List[List[Coord]] = []
    for ring in polygon.interiors:
        scaled_ring = _scaled_ring(list(ring.coords))
        if scaled_ring is not None:
            holes.append(scaled_ring)
    return Polygon(ext, holes), sp0, sp1, scale


def segment_inside_params(polygon: Polygon, p0: Coord, p1: Coord) -> List[Tuple[float, float]]:
    """返回移动线段 p0->p1 位于区域内部/外边界上、且不在洞边界上的参数区间。

    外边界计入区域；洞及其边界不计入。零长度的点接触不产生区间。
    端点分量允许取任意合法有限值（含 ±1e308 量级跨越）：极大模长时按
    比例缩放后再交给 GEOS，避免内部平方/叉积溢出产生空区间或 NaN。
    """
    polygon, p0, p1, _scale = _scaled_polygon_and_segment(polygon, p0, p1)
    if polygon is None:
        return []
    start = Point(p0)
    end = Point(p1)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]

    # 零长线段（静止）：按点归属处理
    if dx == 0.0 and dy == 0.0:
        if point_in_region(polygon, p0):
            return [(0.0, 1.0)]
        return []

    segment = LineString([p0, p1])

    # 1) 与多边形（含外边界，不含洞内部）的一维交集
    inside: List[Tuple[float, float]] = []
    for piece in _line_pieces_on_geometry(segment, polygon):
        ta, tb = _params_of_piece(piece, start, end, dx, dy)
        if tb - ta > _PARAM_EPS:
            inside.append((ta, tb))
    inside = merge_intervals(inside)
    if not inside:
        return []

    # 2) 交集里落在洞边界上的部分必须剔除（Shapely 中洞边界属于 polygon 的边界）
    cuts: List[Tuple[float, float]] = []
    for ring in polygon.interiors:
        for piece in _line_pieces_on_geometry(segment, ring):
            ta, tb = _params_of_piece(piece, start, end, dx, dy)
            if tb - ta > _PARAM_EPS:
                cuts.append((ta, tb))
    cuts = merge_intervals(cuts)

    intervals = _subtract_intervals(inside, cuts) if cuts else inside
    return merge_intervals(intervals)


def point_in_region(polygon: Polygon, p: Coord) -> bool:
    """点是否属于区域：外边界算，洞内部与洞边界不算。"""
    point = Point(p)
    if not polygon.covers(point):
        return False
    for ring in polygon.interiors:
        # touches/intersects 对 LinearRing 均能识别"在洞边界上"
        if ring.intersects(point):
            return False
    return True
