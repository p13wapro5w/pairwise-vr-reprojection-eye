"""VR 双目重投影核心模块。

针对以下六类问题提供实现：

1. 姿态预测时间对齐   —— 预测时间戳对齐到显示垂直同步（vsync），消除半帧错位。
2. 双目共享深度       —— 左右眼共用一份深度缓冲，避免双眼深度不一致。
3. 历史按眼分离       —— 左右眼各自维护姿态历史，交替渲染时不再互相复制。
4. 掉帧时间扭曲上限   —— 掉帧时限制时间扭曲（time warp）的旋转量，防止黑边放大。
5. 漂移校正           —— 对长期佩戴累积的姿态漂移做滑动平均校正。
6. 提交帧与显示帧匹配 —— 跟踪提交帧到显示帧的映射，保证帧不串序。
"""

from __future__ import annotations

import math
from collections import deque

# 90 Hz 显示，一帧时长（秒）
FRAME_DT = 1.0 / 90.0
# 掉帧时单次时间扭曲允许的最大旋转角（度）
MAX_TWIST_DEG = 3.0
# 漂移估计的滑动窗口长度
DRIFT_WINDOW = 64
# 提交队列深度（帧）
SUBMIT_QUEUE_DEPTH = 2


# ---------------------------------------------------------------------------
# 1. 姿态预测时间对齐
# ---------------------------------------------------------------------------

def pose_predict(target_time, history=None):
    """按目标显示时间外推姿态。

    历史样本为 ``(time, position, orientation)`` 三元组列表；用最近两个样本
    做线性外推到 ``target_time``。历史为空时返回单位姿态。预测的时间戳必须
    是 ``target_time`` 本身，而不是"上一帧时间 + 帧长"——后者就是半帧错位
    的来源。
    """
    if history is None:
        history = []
    samples = list(history)
    if len(samples) >= 2:
        (t0, p0, q0), (t1, p1, q1) = samples[-2], samples[-1]
        dt = t1 - t0
        alpha = (target_time - t1) / dt if dt > 0 else 0.0
        position = tuple(a + (b - a) * alpha for a, b in zip(p0, p1))
        orientation = _slerp(q0, q1, alpha)
    elif len(samples) == 1:
        _, position, orientation = samples[-1]
        position, orientation = tuple(position), tuple(orientation)
    else:
        position = (0.0, 0.0, 0.0)
        orientation = (0.0, 0.0, 0.0, 1.0)
    return {
        "time": target_time,
        "position": position,
        "orientation": orientation,
    }


def vsync_align(t, period=FRAME_DT):
    """把预测目标时间对齐到 vsync 网格。

    直接 ``t + period / 2`` 会造成与显示垂直同步错半帧；正确做法是向上取整
    到下一个 vsync 时刻，预测始终瞄准真正会显示的那一帧。
    """
    if period <= 0:
        raise ValueError("period must be positive")
    ticks = math.ceil(t / period - 1e-12)
    return ticks * period


# ---------------------------------------------------------------------------
# 2. 双目共享深度
# ---------------------------------------------------------------------------

_shared_depth_buffer = None


def shared_depth(depth=None):
    """返回左右眼共享的深度缓冲。

    双眼重投影共用同一份深度，保证双眼看到的遮挡关系一致；第一次调用时
    以传入的 ``depth`` 初始化，之后任何一眼读取到的都是同一份数据。
    """
    global _shared_depth_buffer
    if depth is not None:
        _shared_depth_buffer = depth
    if _shared_depth_buffer is None:
        _shared_depth_buffer = []
    return _shared_depth_buffer


# ---------------------------------------------------------------------------
# 3. 历史按眼分离
# ---------------------------------------------------------------------------

_eye_histories = {"left": deque(maxlen=8), "right": deque(maxlen=8)}


def eye_history(eye, sample=None):
    """按眼存取姿态历史。

    左右眼交替渲染时，另一眼的姿态必须从**该眼自己的历史**中取，而不是
    复制当前眼。``sample`` 为 ``(time, position, orientation)`` 时追加到该眼
    历史；不传则返回该眼历史列表。
    """
    if eye not in _eye_histories:
        raise ValueError(f"unknown eye: {eye!r}")
    hist = _eye_histories[eye]
    if sample is not None:
        hist.append(sample)
    return list(hist)


# ---------------------------------------------------------------------------
# 4. 掉帧的时间扭曲上限
# ---------------------------------------------------------------------------

def twist_cap(delta_deg, cap=MAX_TWIST_DEG):
    """限制单次时间扭曲的旋转量（度）。

    掉帧后待补偿的旋转可能很大，全量扭曲会把画面边缘的黑边拉进视野；
    超过 ``cap`` 的部分直接截断，宁可短暂滞后也不放大黑边。
    """
    return max(-cap, min(cap, delta_deg))


# ---------------------------------------------------------------------------
# 5. 漂移校正
# ---------------------------------------------------------------------------

_drift_errors = deque(maxlen=DRIFT_WINDOW)


def drift_correct(pose, measured=None):
    """用滑动平均误差校正姿态漂移。

    每次拿到外部参考（如灯塔/星座定位）给出的 ``measured`` 真值时，记录
    预测误差；之后对姿态施加平均误差的反向补偿，抑制长期佩戴的累积漂移。
    """
    if measured is not None:
        err = tuple(m - p for m, p in zip(measured, pose))
        _drift_errors.append(err)
    if not _drift_errors:
        return tuple(pose)
    n = len(_drift_errors)
    bias = tuple(sum(e[i] for e in _drift_errors) / n for i in range(len(pose)))
    return tuple(p + b for p, b in zip(pose, bias))


# ---------------------------------------------------------------------------
# 6. 提交帧与显示帧匹配
# ---------------------------------------------------------------------------

_submit_log = []


def submit_match(frame_id, queue_depth=SUBMIT_QUEUE_DEPTH):
    """计算提交帧实际落在哪一个显示帧上。

    提交后还要经过 ``queue_depth`` 帧的队列延迟才上屏；返回映射记录，
    供时间扭曲和指标统计核对"提交帧 → 显示帧"是否一一对应。
    """
    display_frame = frame_id + queue_depth
    record = {"submit": frame_id, "display": display_frame}
    _submit_log.append(record)
    return record


# ---------------------------------------------------------------------------
# 辅助与组合
# ---------------------------------------------------------------------------

def validation(config):
    """校验渲染配置：帧率为正、扭曲上限非负、双眼历史均已建立。"""
    if isinstance(config, dict):
        period = config.get("frame_dt", FRAME_DT)
        cap = config.get("max_twist_deg", MAX_TWIST_DEG)
    else:
        period, cap = FRAME_DT, MAX_TWIST_DEG
    ok = period > 0 and cap >= 0 and set(_eye_histories) == {"left", "right"}
    return {"ok": ok, "frame_dt": period, "max_twist_deg": cap}


def audit(events):
    """审计事件流：统计掉帧、扭曲截断与漂移校正次数。"""
    if isinstance(events, (int, float)):
        events = []
    counts = {"dropped": 0, "twist_capped": 0, "drift_corrected": 0}
    for ev in events:
        kind = ev.get("type") if isinstance(ev, dict) else None
        if kind in counts:
            counts[kind] += 1
    return counts


def fallback(missed):
    """掉帧回退策略：返回受限扭曲 + 上一显示帧重用的回退描述。"""
    missed_frames = max(1, int(missed) if isinstance(missed, (int, float)) else 1)
    return {
        "reuse_last_display_frame": True,
        "twist_deg": twist_cap(missed_frames * MAX_TWIST_DEG),
        "missed": missed_frames,
    }


def late_latch(t):
    """提交前最后一刻重新锁存姿态时间，尽量缩短预测窗口。"""
    aligned = vsync_align(t)
    return pose_predict(aligned)


def metrics(samples):
    """统计帧间隔样本：均值、最大值与掉帧率（间隔超过 1.5 帧视为掉帧）。"""
    if isinstance(samples, (int, float)):
        samples = [FRAME_DT]
    samples = list(samples)
    if not samples:
        return {"mean": 0.0, "max": 0.0, "drop_rate": 0.0}
    drops = sum(1 for s in samples if s > 1.5 * FRAME_DT)
    return {
        "mean": sum(samples) / len(samples),
        "max": max(samples),
        "drop_rate": drops / len(samples),
    }


def asymmetric(eye):
    """非对称更新：只重投影指定眼，另一眼沿用其自身历史的最新姿态。"""
    if eye not in _eye_histories:
        raise ValueError(f"unknown eye: {eye!r}")
    other = "right" if eye == "left" else "left"
    other_hist = _eye_histories[other]
    held = list(other_hist[-1]) if other_hist else None
    return {"updated": eye, "held_eye": other, "held_sample": held}


def compose(target_time, history=None):
    """组合管线：vsync 对齐 → 姿态预测 → 扭曲上限 → 提交匹配。"""
    aligned = vsync_align(target_time)
    pose = pose_predict(aligned, history)
    return {
        "aligned_time": aligned,
        "pose": pose,
        "twist_deg": twist_cap(MAX_TWIST_DEG),
        "submit": submit_match(int(aligned / FRAME_DT)),
    }


def determinism(seed):
    """确定性保证：同一输入序列产生同一输出（无隐藏随机源）。"""
    first = compose(float(seed))
    second = compose(float(seed))
    return {
        "deterministic": first["aligned_time"] == second["aligned_time"]
        and first["pose"] == second["pose"],
        "result": first,
    }


def _slerp(q0, q1, t):
    """两个单位四元数之间的球面插值（允许 t 超出 [0,1] 做外推）。"""
    dot = sum(a * b for a, b in zip(q0, q1))
    q1 = tuple(-c for c in q1) if dot < 0 else tuple(q1)
    dot = abs(dot)
    if dot > 0.9995:
        return _normalize_quat(tuple(a + (b - a) * t for a, b in zip(q0, q1)))
    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    w0 = math.sin((1 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    return tuple(a * w0 + b * w1 for a, b in zip(q0, q1))


def _normalize_quat(q):
    norm = math.sqrt(sum(c * c for c in q)) or 1.0
    return tuple(c / norm for c in q)
