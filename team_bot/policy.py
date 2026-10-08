from __future__ import annotations

import heapq
import math
import sys
import time
from typing import Any


# Feature switches (set a value to False to fall back to the older behaviour)

FEATURES = {
    "joint_kick": True,        # evaluate kicks from the post-move origin, choose (move, kick) jointly
    "fine_verify": True,       # re-check top kick candidates at the engine's ball resolution
    "path_race": True,         # loose-ball race uses obstacle-aware path distance
    "nav_progress": True,      # carry progress = drop in path distance (no hard "never backwards")
    "shot_map": True,          # carry toward cells that have a scoring lane
    "defend_worst_case": True, # defend against the opponent's best post-move shot
    "loose_commit": True,      # use loose_ball_steps to break stationary-ball standoffs
    "loop_detect": True,       # repeated-state detector switches tactic variant
}
F = FEATURES

TIME_BUDGET = 0.80            # seconds per decision (hard limit is 2.0 s)
SHOT_SLICE = 0.08             # seconds of shot-map building allowed per decision


# Constants

DIRECTION_VECTORS = {
    "STAY": (0, 0),
    "UP": (0, 1),
    "UP_RIGHT": (1, 1),
    "RIGHT": (1, 0),
    "DOWN_RIGHT": (1, -1),
    "DOWN": (0, -1),
    "DOWN_LEFT": (-1, -1),
    "LEFT": (-1, 0),
    "UP_LEFT": (-1, 1),
}
MOVES = list(DIRECTION_VECTORS)
ALL_DIRS = [name for name in MOVES if name != "STAY"]
UNIT = {
    name: ((0.0, 0.0) if name == "STAY" else (vx / math.hypot(vx, vy), vy / math.hypot(vx, vy)))
    for name, (vx, vy) in DIRECTION_VECTORS.items()
}

BALL_RADIUS = 1.5
BALL_SPEED = 8.0
KICK_DISTANCES = (32.0, 64.0, 96.0)
POSSESSION_RADIUS = 5.0
TACKLE_CONTACT = 6.15
SAFE_CARRY_STEPS = 3          # opponent cannot tackle until possession_steps >= 3
FORCED_RELEASE_STEPS = 10     # engine forces a power-1 straight kick when this is reached
MAXIMUM_GOALS = 7             # total goals (both players) that end the match
PLAN_SUBSTEP = 2.0            # coarse planning resolution for the ball
FINE_SUBSTEP = 0.675          # the engine's real ball resolution

GOAL_VALUE = 100.0
RACE_MARGIN = 0.0
BIAS_CYCLE = (10.0, -1.0, 4.0)   # dribble-vs-kick tactic, cycled by goals conceded + loop variant
SHOOT_NOW = 60.0
LOOKAHEAD = 3
URGENCY_SHOOT = 18.0
RACE_WEIGHT = 12.0
BOUNCE_BACK_PENALTY = 30.0
MOUTH_BONUS = 4.0
PROG_SCALE = 4.0              # weight of the navigation-progress term when carrying
MOVE_PROG_BONUS = 1.5         # small preference for kick-moves that also advance
VERIFY_TOP = 4                # kick (move, direction) pairs re-checked at fine resolution
LOOSE_COMMIT_STEPS = 8        # loose_ball_steps at which we stop hedging and go for the ball
SHOT_CELL = 5.0

_GRID_CACHE: dict = {}
_FIELD_CACHE: dict = {}
_SHOT_CACHE: dict = {}


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def direction_toward(dx: float, dy: float, dead_zone: float = 1.0) -> str:
    horizontal = "" if abs(dx) <= dead_zone else ("RIGHT" if dx > 0 else "LEFT")
    vertical = "" if abs(dy) <= dead_zone else ("UP" if dy > 0 else "DOWN")
    return f"{vertical}_{horizontal}" if vertical and horizontal else vertical or horizontal or "STAY"



# Per-observation context

class Ctx:
    def __init__(self, observation: dict[str, Any], deadline: float | None = None) -> None:
        state = observation["state"]
        field = state["field"]
        self.deadline = deadline if deadline is not None else time.perf_counter() + TIME_BUDGET
        self.player_id = observation["player_id"]
        self.opponent_id = observation["opponent_id"]
        self.W = float(field["width"])
        self.H = float(field["height"])
        self.goal_w = float(field.get("goal_width", 36.0))
        self.goal_l = (self.W - self.goal_w) / 2
        self.goal_r = self.goal_l + self.goal_w
        self.radius = float(field.get("player_radius", 3.0))
        self.speed = float(field.get("player_speed", 4.0))
        self.sign = 1.0 if observation["attack_direction"] == "UP" else -1.0
        self.attack = observation["attack_direction"]
        me = state["players"][self.player_id]
        opp = state["players"][self.opponent_id]
        self.me = (float(me["x"]), float(me["y"]))
        self.opp = (float(opp["x"]), float(opp["y"]))
        ball = state["ball"]
        self.ball = ball
        self.bx, self.by = float(ball["x"]), float(ball["y"])
        self.status = ball["status"]
        self.possession = ball["possession"]
        self.poss_steps = int(ball.get("possession_steps", 0))
        self.loose_steps = int(ball.get("loose_ball_steps", 0))
        velocity = ball.get("velocity", {}) or {}
        self.bvx, self.bvy = float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0))
        self.ball_remaining = float(ball.get("remaining_kick_distance", 0.0))
        speed = math.hypot(self.bvx, self.bvy)
        self.ball_speed = speed if speed > 1e-6 else BALL_SPEED
        self.obstacles = [
            (float(o["x"]), float(o["y"]), float(o["x"]) + float(o["width"]), float(o["y"]) + float(o["height"]))
            for o in state.get("obstacles", [])
        ]
        self.layout = tuple(self.obstacles)
        self.iteration = int(state.get("iteration", 0))
        score = state.get("score") or {}
        self.score = score
        self.conceded = int(score.get(self.opponent_id, 0))
        self.lead = int(score.get(self.player_id, 0)) - self.conceded
        self.goals_left = max(0, MAXIMUM_GOALS - sum(int(v) for v in score.values()))
        self.dribble_bias = BIAS_CYCLE[0]
        self.max_iter = max(1, int(state.get("maximum_iterations", 400)))
        self.my_goal_y = 0.0 if self.sign > 0 else self.H
        self.opp_goal_y = self.H if self.sign > 0 else 0.0
        self.my_goal = (self.W / 2, self.my_goal_y)
        self.opp_goal = (self.W / 2, self.opp_goal_y)
        self.no_kick = False
        self.trace_cache: dict = {}

    def out_of_time(self) -> bool:
        return time.perf_counter() > self.deadline

    # geometry helpers
    def circle_hits_obstacle(self, x: float, y: float, r: float) -> bool:
        for x0, y0, x1, y1 in self.obstacles:
            cx = x0 if x < x0 else x1 if x > x1 else x
            cy = y0 if y < y0 else y1 if y > y1 else y
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                return True
        return False

    def player_ok(self, x: float, y: float, margin: float = 0.1) -> bool:
        r = self.radius + margin
        if x - r < 0 or x + r > self.W or y - r < 0 or y + r > self.H:
            return False
        return not self.circle_hits_obstacle(x, y, r)

    def step_pos(self, pos: tuple[float, float], move: str) -> tuple[float, float]:
        ux, uy = UNIT[move]
        return pos[0] + ux * self.speed, pos[1] + uy * self.speed

    def valid_moves(self, pos: tuple[float, float] | None = None, margin: float = 0.1) -> list[str]:
        pos = pos or self.me
        out = []
        for move in MOVES:
            if move == "STAY":
                continue
            x, y = self.step_pos(pos, move)
            if self.player_ok(x, y, margin):
                out.append(move)
        return out

    def segment_clear(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(dist / 2.0))
        for i in range(1, n):
            t = i / n
            if not self.player_ok(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.1):
                return False
        return True

    # flood-fill navigation fields
    def free_grid(self):
        cached = _GRID_CACHE.get(self.layout)
        if cached is not None:
            return cached
        cell = 2.0
        nx, ny = int(self.W / cell), int(self.H / cell)
        free = [[self.player_ok((ix + 0.5) * cell, (iy + 0.5) * cell, 0.6) for iy in range(ny)] for ix in range(nx)]
        if len(_GRID_CACHE) > 8:
            _GRID_CACHE.clear()
        _GRID_CACHE[self.layout] = (free, cell, nx, ny)
        return _GRID_CACHE[self.layout]

    def nav_field(self, target: tuple[float, float], stops=None, full: bool = False):
        free, cell, nx, ny = self.free_grid()
        tx = int(_clamp(target[0] / cell, 0, nx - 1))
        ty = int(_clamp(target[1] / cell, 0, ny - 1))
        remaining = None
        if not full:
            remaining = {
                (int(_clamp(p[0] / cell, 0, nx - 1)), int(_clamp(p[1] / cell, 0, ny - 1)))
                for p in (stops or [self.me])
            }
        dist: dict = {}
        heap: list = []
        seeds = [(tx, ty)]
        if not free[tx][ty]:
            seeds = [
                (tx + dx, ty + dy)
                for dx in range(-4, 5)
                for dy in range(-4, 5)
                if 0 <= tx + dx < nx and 0 <= ty + dy < ny and free[tx + dx][ty + dy]
            ]
        for s in seeds:
            dist[s] = 0.0
            heapq.heappush(heap, (0.0, s))
        limit = None
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, 1e18):
                continue
            if limit is not None and d > limit:
                break
            if remaining is not None and limit is None:
                remaining.discard(node)
                if not remaining:
                    limit = d + 4.0
            ix, iy = node
            for dx, dy, c in steps:
                jx, jy = ix + dx, iy + dy
                if jx < 0 or jy < 0 or jx >= nx or jy >= ny or not free[jx][jy]:
                    continue
                nd = d + c
                if nd < dist.get((jx, jy), 1e18):
                    dist[(jx, jy)] = nd
                    heapq.heappush(heap, (nd, (jx, jy)))
        return dist, cell, nx, ny

    def cached_field(self, target: tuple[float, float]):
        key = (self.layout, round(target[0] / 2.0), round(target[1] / 2.0))
        field = _FIELD_CACHE.get(key)
        if field is None:
            if len(_FIELD_CACHE) > 60:
                _FIELD_CACHE.clear()
            field = self.nav_field(target, full=True)
            _FIELD_CACHE[key] = field
        return field

    def nav_cost(self, field, pos: tuple[float, float], fallback_target: tuple[float, float]) -> float:
        dist, cell, nx, ny = field
        ix = int(_clamp(pos[0] / cell, 0, nx - 1))
        iy = int(_clamp(pos[1] / cell, 0, ny - 1))
        best = dist.get((ix, iy))
        if best is None:
            best = 1e4 + math.hypot(pos[0] - fallback_target[0], pos[1] - fallback_target[1])
        return best * cell


def navigate(ctx: Ctx, target: tuple[float, float], stop_distance: float = 0.0, allow_stay: bool = True,
             field=None, cache: bool = False) -> str:
    me = ctx.me
    dx, dy = target[0] - me[0], target[1] - me[1]
    distance = math.hypot(dx, dy)
    if distance <= max(stop_distance, 0.3) and allow_stay:
        return "STAY"
    direct = ctx.segment_clear(me, target) if distance < 160 else False
    if direct:
        candidates = []
        for move in MOVES:
            if move == "STAY":
                continue
            x, y = ctx.step_pos(me, move)
            if not ctx.player_ok(x, y):
                continue
            candidates.append((math.hypot(target[0] - x, target[1] - y), move))
        if candidates:
            best = min(candidates)
            if allow_stay and best[0] >= distance - 0.2:
                return "STAY"
            return best[1]
    if field is None:
        field = ctx.cached_field(target) if cache else ctx.nav_field(target)
    base = ctx.nav_cost(field, me, target)
    scored = []
    for move in MOVES:
        if move == "STAY":
            continue
        x, y = ctx.step_pos(me, move)
        if not ctx.player_ok(x, y):
            continue
        scored.append((ctx.nav_cost(field, (x, y), target) + 0.01 * math.hypot(target[0] - x, target[1] - y), move))
    if not scored:
        return "STAY"
    best = min(scored)
    if allow_stay and best[0] >= base and distance < 6:
        return "STAY"
    return best[1]



# Ball physics

def ball_trace(ctx: Ctx, x: float, y: float, ux: float, uy: float, distance: float, max_steps: int = 14,
               ball_speed: float | None = None, substep: float = PLAN_SUBSTEP):
    """Simulate a ball. Returns (points, goal); points are (t, x, y, travelled) and goal is
    ('top'|'bottom', t, travelled) or None. t is the 1-based simulation step."""
    speed = ball_speed or ctx.ball_speed
    W, H, r = ctx.W, ctx.H, BALL_RADIUS
    gl, gr = ctx.goal_l, ctx.goal_r
    obstacles = ctx.obstacles
    vx, vy = ux, uy
    remaining = distance
    travelled = 0.0
    points = []
    t = 0
    while remaining > 1e-9 and t < max_steps:
        t += 1
        travel = min(speed, remaining)
        subs = max(1, math.ceil(travel / substep))
        sd = travel / subs
        for _ in range(subs):
            px, py = x, y
            cx, cy = px + vx * sd, py + vy * sd
            if gl <= cx <= gr:
                if cy + r >= H:
                    points.append((t, cx, cy, travelled + sd))
                    return points, ("top", t, travelled + sd)
                if cy - r <= 0:
                    points.append((t, cx, cy, travelled + sd))
                    return points, ("bottom", t, travelled + sd)
            if cx - r < 0 or cx + r > W:
                vx = -vx
                cx = _clamp(cx, r, W - r)
            if cy - r < 0 or cy + r > H:
                vy = -vy
                cy = _clamp(cy, r, H - r)
            for x0, y0, x1, y1 in obstacles:
                qx = x0 if cx < x0 else x1 if cx > x1 else cx
                qy = y0 if cy < y0 else y1 if cy > y1 else cy
                if (cx - qx) ** 2 + (cy - qy) ** 2 < r * r:
                    left, right, bottom, top = x0 - r, x1 + r, y0 - r, y1 + r
                    crossed_x = px <= left or px >= right
                    crossed_y = py <= bottom or py >= top
                    if crossed_x:
                        vx = -vx
                    if crossed_y:
                        vy = -vy
                    if not crossed_x and not crossed_y:
                        vx, vy = -vx, -vy
                    cx, cy = px + vx * 0.05, py + vy * 0.05
                    break
            x, y = cx, cy
            travelled += sd
            remaining -= sd
            points.append((t, x, y, travelled))
    return points, None


def _kick_trace(ctx: Ctx, sx: float, sy: float, direction: str, fine: bool):
    key = (round(sx, 2), round(sy, 2), direction, fine)
    cached = ctx.trace_cache.get(key)
    if cached is None:
        ux, uy = UNIT[direction]
        cached = ball_trace(ctx, sx, sy, ux, uy, KICK_DISTANCES[-1], ball_speed=BALL_SPEED,
                            substep=FINE_SUBSTEP if fine else PLAN_SUBSTEP)
        ctx.trace_cache[key] = cached
    return cached



# Kick evaluation

def _race_time(ctx: Ctx, pos: tuple[float, float], target: tuple[float, float]) -> float:
    return max(0.0, math.hypot(pos[0] - target[0], pos[1] - target[1]) - POSSESSION_RADIUS) / ctx.speed


def evaluate_kicks(ctx: Ctx, pos: tuple[float, float], opp: tuple[float, float], directions: list[str],
                   danger: float = 1.0, t_offset: float = 0.0, fine: bool = False):
    results = []
    sign = ctx.sign
    for direction in directions:
        ux, uy = UNIT[direction]
        clearance = ctx.radius + BALL_RADIUS + 0.05
        sx, sy = pos[0] + ux * clearance, pos[1] + uy * clearance
        points, goal = _kick_trace(ctx, sx, sy, direction, fine)
        slacks = []
        first_reach = None
        running = -1e9
        for index, (t, px, py, travelled) in enumerate(points):
            slack = 4.5 + ctx.speed * (t + t_offset) - math.hypot(opp[0] - px, opp[1] - py)
            running = max(running, slack)
            slacks.append(running)
            if first_reach is None and slack >= 0.0:
                first_reach = index
        for power in (1, 2, 3):
            limit = KICK_DISTANCES[power - 1]
            end_index = len(points) - 1
            for index, point in enumerate(points):
                if point[3] >= limit - 1e-6:
                    end_index = index
                    break
            scored_here = goal is not None and goal[2] <= limit + 1e-6
            if scored_here:
                end_index = len(points) - 1
            slack_max = slacks[end_index] if slacks else -99.0
            p_int = _clamp((slack_max + 5.0) / 10.0, 0.0, 1.0)
            if first_reach is not None and first_reach <= end_index:
                ip = points[first_reach]
                my_dist_to_goal = abs(ip[2] - ctx.my_goal_y)
                v_int = -6.0 - 14.0 * danger * (1.0 - my_dist_to_goal / ctx.H)
            else:
                v_int = -10.0
            if scored_here:
                mine = (goal[0] == "top") == (sign > 0)
                if mine:
                    value = (1 - p_int) * GOAL_VALUE + p_int * v_int
                else:
                    value = -GOAL_VALUE
                results.append((value, direction, power, {"goal": mine, "p_int": p_int}))
                continue
            final = points[end_index] if points else (0, sx, sy, 0.0)
            fx, fy = final[1], final[2]
            t_me = _race_time(ctx, pos, (fx, fy)) + 0.5
            t_opp = _race_time(ctx, opp, (fx, fy))
            progress = sign * (fy - ctx.H / 2) / (ctx.H / 2)
            v_end = RACE_WEIGHT * _clamp((t_opp - t_me) / 3.0, -1.0, 1.0) + 9.0 * progress
            if progress > 0.4:
                v_end += MOUTH_BONUS * (1.0 - min(1.0, abs(fx - ctx.W / 2) / (ctx.W / 2)))
            for _t, bx, by, travelled_d in points[: end_index + 1]:
                if travelled_d > 8.0 and math.hypot(bx - pos[0], by - pos[1]) < 5.5:
                    v_end -= BOUNCE_BACK_PENALTY
                    break
            value = (1 - p_int) * v_end + p_int * v_int
            results.append((value, direction, power, {"goal": False, "p_int": p_int}))
    return results



# Shot map: which cells have a clean scoring lane (ignoring the opponent), built lazily

def _shot_state(ctx: Ctx):
    key = (ctx.layout, ctx.sign)
    state = _SHOT_CACHE.get(key)
    if state is None:
        cells = []
        nx, ny = int(ctx.W / SHOT_CELL), int(ctx.H / SHOT_CELL)
        for ix in range(nx):
            for iy in range(ny):
                cx, cy = (ix + 0.5) * SHOT_CELL, (iy + 0.5) * SHOT_CELL
                if ctx.sign * (cy - ctx.H / 2) < 0:
                    continue  # only the attacking half matters
                if ctx.player_ok(cx, cy, 0.6):
                    cells.append((abs(cy - ctx.opp_goal_y), ix, iy))
        cells.sort()
        state = {"todo": [(ix, iy) for _, ix, iy in cells], "i": 0, "shots": {}}
        if len(_SHOT_CACHE) > 6:
            _SHOT_CACHE.clear()
        _SHOT_CACHE[key] = state
    stop = min(ctx.deadline - 0.2, time.perf_counter() + SHOT_SLICE)
    todo = state["todo"]
    clearance = ctx.radius + BALL_RADIUS + 0.05
    while state["i"] < len(todo) and time.perf_counter() < stop:
        ix, iy = todo[state["i"]]
        state["i"] += 1
        cx, cy = (ix + 0.5) * SHOT_CELL, (iy + 0.5) * SHOT_CELL
        lanes = []
        for direction in ALL_DIRS:
            ux, uy = UNIT[direction]
            points, goal = ball_trace(ctx, cx + ux * clearance, cy + uy * clearance, ux, uy,
                                      KICK_DISTANCES[-1], ball_speed=BALL_SPEED)
            if goal is not None and goal[2] <= KICK_DISTANCES[-1] and (goal[0] == "top") == (ctx.sign > 0):
                lanes.append(direction)
        if lanes:
            state["shots"][(ix, iy)] = lanes
    return state


def shot_target(ctx: Ctx):
    state = _shot_state(ctx)
    shots = state["shots"]
    if not shots:
        return None
    mine = (int(ctx.me[0] / SHOT_CELL), int(ctx.me[1] / SHOT_CELL))
    if mine in shots:
        return None
    field = ctx.nav_field(ctx.me, full=True)
    best, best_score = None, 1e18
    for (ix, iy), lanes in shots.items():
        c = ((ix + 0.5) * SHOT_CELL, (iy + 0.5) * SHOT_CELL)
        d = ctx.nav_cost(field, c, c)
        if d >= 1e4:
            continue
        crowd = max(0.0, 25.0 - math.hypot(c[0] - ctx.opp[0], c[1] - ctx.opp[1])) * 0.6
        score = d + crowd - 3.0 * len(lanes)
        if score < best_score:
            best, best_score = c, score
    return best


def make_progress(ctx: Ctx):
    """Returns f(pos) -> progress (about -1.5..1.5 per step) toward the carry target."""
    me, speed, sign = ctx.me, ctx.speed, ctx.sign
    if not F["nav_progress"]:
        return lambda pos: sign * (pos[1] - me[1]) / speed
    target = shot_target(ctx) if F["shot_map"] else None
    if target is None:
        target = ctx.opp_goal
    field = ctx.cached_field(target)
    base = ctx.nav_cost(field, me, target)
    return lambda pos: _clamp((base - ctx.nav_cost(field, pos, target)) / speed, -1.5, 1.5)



# Tactical logic

def _forward_dirs(ctx: Ctx, wide: bool = False) -> list[str]:
    a = ctx.attack
    base = [a, f"{a}_LEFT", f"{a}_RIGHT"]
    if wide:
        base += ["LEFT", "RIGHT"]
    return base


def _best_kick(ctx: Ctx, pos, opp, directions) -> tuple[float, str, int]:
    results = evaluate_kicks(ctx, pos, opp, directions)
    best = max(results, key=lambda item: (item[0], item[2]))
    return best[0], best[1], best[2]


def best_joint_kick(ctx: Ctx, directions: list[str], prog_fn):
    me, opp = ctx.me, ctx.opp
    options = [("STAY", me)]
    for move in ctx.valid_moves():
        q = ctx.step_pos(me, move)
        if math.hypot(q[0] - opp[0], q[1] - opp[1]) < 2 * ctx.radius + 0.5:
            continue
        options.append((move, q))
    cands = []
    for move, q in options:
        bonus = (0.6 if move == "STAY" else 0.0) + MOVE_PROG_BONUS * prog_fn(q)
        for value, direction, power, _info in evaluate_kicks(ctx, q, opp, directions):
            cands.append((value + bonus, move, direction, power, q, bonus))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[3]), reverse=True)
    if not F["fine_verify"]:
        c = cands[0]
        return c[0], c[1], c[2], c[3]
    verified, seen = [], set()
    for c in cands:
        key = (c[1], c[2])
        if key in seen:
            continue
        seen.add(key)
        for value, direction, power, _info in evaluate_kicks(ctx, c[4], opp, [c[2]], fine=True):
            verified.append((value + c[5], c[1], direction, power))
        if len(seen) >= VERIFY_TOP or ctx.out_of_time():
            break
    return max(verified, key=lambda v: (v[0], v[3]))


def _approach(pos, target, speed):
    d = math.hypot(target[0] - pos[0], target[1] - pos[1])
    if d <= speed:
        return target
    return pos[0] + (target[0] - pos[0]) / d * speed, pos[1] + (target[1] - pos[1]) / d * speed


def urgency(ctx: Ctx) -> float:
    elapsed = ctx.iteration / ctx.max_iter
    if ctx.lead < 0:
        boost = 0.3 if (elapsed > 0.6 and ctx.goals_left <= 2) else 0.0
        return min(1.5, 0.6 * (-ctx.lead) + 0.6 * elapsed + boost)
    if ctx.lead == 0:
        return max(0.0, (elapsed - 0.5) / 0.5) * 0.5
    return -min(0.6, 0.2 * ctx.lead + 0.3 * elapsed)


def _follow(ctx: Ctx, opp, pos, steps: float):
    d = math.hypot(opp[0] - pos[0], opp[1] - pos[1])
    keep = max(2 * ctx.radius, d - ctx.speed * steps)
    if d <= keep or d < 1e-9:
        return opp
    return pos[0] + (opp[0] - pos[0]) / d * keep, pos[1] + (opp[1] - pos[1]) / d * keep


def attack_with_ball(ctx: Ctx) -> dict[str, Any]:
    me, opp, sign = ctx.me, ctx.opp, ctx.sign
    s = ctx.poss_steps
    kick_dirs = _forward_dirs(ctx, wide=True)
    reach = ctx.speed + TACKLE_CONTACT
    opp_dist = math.hypot(opp[0] - me[0], opp[1] - me[1])
    must_kick = s >= FORCED_RELEASE_STEPS - 1 or (s >= SAFE_CARRY_STEPS and opp_dist <= reach + 4.0)

    prog_fn = make_progress(ctx)

    if F["joint_kick"]:
        joint = best_joint_kick(ctx, kick_dirs, prog_fn)
    else:
        value, direction, power = _best_kick(ctx, me, opp, kick_dirs)
        joint = (value, "STAY", direction, power)
    if joint is None:
        return {"move": navigate(ctx, ctx.opp_goal, allow_stay=False, cache=True)}
    now_value, now_move, now_dir, now_power = joint

    # situational aggression 
    urg = urgency(ctx)
    bias = ctx.dribble_bias
    progress_weight = 0.4 + 1.0 * max(0.0, urg)
    shoot_now = _clamp(SHOOT_NOW - URGENCY_SHOOT * urg, 35.0, 80.0)
    opp_behind = sign * (opp[1] - me[1]) < -6.0
    opp_deep = abs(opp[1] - ctx.opp_goal_y) > ctx.H * 0.55
    if opp_behind:
        bias += 4.0
    if opp_deep and s <= 1:
        shoot_now -= 10.0

    def kick_action(move: str) -> dict[str, Any]:
        return {"move": move, "kick": {"direction": now_dir, "power": now_power}}

    if now_value >= shoot_now and not ctx.no_kick:
        return kick_action(now_move)

    best_move, best_value = None, -1e9
    if not must_kick:
        for move in ctx.valid_moves():
            if ctx.out_of_time():
                break
            ux, uy = UNIT[move]
            if not F["nav_progress"] and uy * sign < -0.5:
                continue  
            first = ctx.step_pos(me, move)
            pos, best_k = me, None
            for k in range(1, LOOKAHEAD + 1):
                pos = ctx.step_pos(pos, move)
                if not ctx.player_ok(*pos) or s + k > FORCED_RELEASE_STEPS - 2:
                    break
                opp_k = _follow(ctx, opp, pos, k)
                gap = math.hypot(opp_k[0] - pos[0], opp_k[1] - pos[1])
                if s + k >= SAFE_CARRY_STEPS and gap <= TACKLE_CONTACT + 0.5:
                    break  
                value, _, _ = _best_kick(ctx, pos, opp_k, kick_dirs)
                value *= 0.93 ** k
                if best_k is None or value > best_k:
                    best_k = value
            if best_k is None:
                continue
            if F["nav_progress"]:
                value = best_k + progress_weight * PROG_SCALE * prog_fn(first)
            else:
                value = best_k + progress_weight * (sign * uy)
            if value > best_value:
                best_move, best_value = move, value
    if best_move is not None and (ctx.no_kick or best_value > now_value - bias):
        return {"move": best_move}
    if ctx.no_kick and not must_kick:
        move = navigate(ctx, ctx.opp_goal, allow_stay=False, cache=True)
        if ctx.player_ok(*ctx.step_pos(me, move)):
            return {"move": move}
    if not F["joint_kick"]:
        move = navigate(ctx, ctx.opp_goal, allow_stay=False, cache=True)
        return kick_action(move if ctx.player_ok(*ctx.step_pos(me, move)) else "STAY")
    return kick_action(now_move)


def defend_goal_side(ctx: Ctx) -> tuple[float, float]:
    ox, oy = ctx.opp
    gx, gy = ctx.my_goal
    d = math.hypot(gx - ox, gy - oy) or 1.0
    ux, uy = (gx - ox) / d, (gy - oy) / d
    gap = min(9.0, d * 0.5)
    return ox + ux * gap, oy + uy * gap


def chase_loose_ball(ctx: Ctx) -> str:
    me, opp = ctx.me, ctx.opp
    if ctx.status == "moving" and ctx.ball_remaining > 0:
        speed = ctx.ball_speed
        vx, vy = ctx.bvx / speed, ctx.bvy / speed
        points, goal = ball_trace(ctx, ctx.bx, ctx.by, vx, vy, ctx.ball_remaining, ball_speed=speed)
        best = None
        gap_best = None
        for t, px, py, travelled in points:
            gap = math.hypot(px - me[0], py - me[1]) - 4.5 - ctx.speed * t
            if gap <= 0:
                best = (px, py, t)
                break
            if gap_best is None or gap < gap_best[0]:
                gap_best = (gap, px, py, t)
        if best is not None:
            return navigate(ctx, (best[0], best[1]), stop_distance=0.5)
        final = points[-1] if points else (0, ctx.bx, ctx.by, 0)
        if gap_best is not None and goal is not None and ((goal[0] == "top") != (ctx.sign > 0)):
            return navigate(ctx, (gap_best[1], gap_best[2]), stop_distance=0.5)
        return navigate(ctx, (final[1], final[2]), stop_distance=1.0)
    ball = (ctx.bx, ctx.by)
    field = None
    if F["path_race"]:
        field = ctx.nav_field(ball, stops=[me, opp])
        t_me = max(0.0, ctx.nav_cost(field, me, ball) - POSSESSION_RADIUS) / ctx.speed
        t_opp = max(0.0, ctx.nav_cost(field, opp, ball) - POSSESSION_RADIUS) / ctx.speed
    else:
        t_me = _race_time(ctx, me, ball)
        t_opp = _race_time(ctx, opp, ball)
    commit = F["loose_commit"] and ctx.loose_steps >= LOOSE_COMMIT_STEPS and t_me <= t_opp + 1.0
    if t_opp < t_me - RACE_MARGIN and not commit:
        gx, gy = ctx.my_goal
        d = math.hypot(gx - ball[0], gy - ball[1]) or 1.0
        pull = (ball[0] + (gx - ball[0]) / d * 8.0, ball[1] + (gy - ball[1]) / d * 8.0)
        return defensive_move(ctx, ball, t_offset=t_opp, pull=pull)
    return navigate(ctx, ball, stop_distance=0.0, allow_stay=False, field=field)


def flip_ctx(ctx: Ctx) -> Ctx:
    f = Ctx.__new__(Ctx)
    f.__dict__.update(ctx.__dict__)
    f.sign = -ctx.sign
    f.attack = "DOWN" if ctx.attack == "UP" else "UP"
    f.me, f.opp = ctx.opp, ctx.me
    f.my_goal_y, f.opp_goal_y = ctx.opp_goal_y, ctx.my_goal_y
    f.my_goal, f.opp_goal = ctx.opp_goal, ctx.my_goal
    return f


def _threat(fctx: Ctx, shooter, defender, t_offset: float = 0.0) -> float:
    results = evaluate_kicks(fctx, shooter, defender, ALL_DIRS, t_offset=t_offset)
    return max(item[0] for item in results)


def defensive_move(ctx: Ctx, shooter, t_offset: float = 0.0, pull=None, spread: bool = False) -> str:
    fctx = flip_ctx(ctx)
    me = ctx.me
    shooters = [shooter]
    if spread and F["defend_worst_case"]:
        for move in ALL_DIRS:
            q = ctx.step_pos(shooter, move)
            if ctx.player_ok(*q):
                shooters.append(q)
    best = None
    for move in MOVES:
        q = me if move == "STAY" else ctx.step_pos(me, move)
        if move != "STAY" and not ctx.player_ok(*q):
            continue
        if best is not None and ctx.out_of_time():
            break
        threat = max(_threat(fctx, sp, q, t_offset) for sp in shooters)
        tie = math.hypot(q[0] - pull[0], q[1] - pull[1]) if pull else 0.0
        key = (round(threat / 3.0), tie)
        if best is None or key < best[0]:
            best = (key, move)
    return best[1] if best else "STAY"


def press_carrier(ctx: Ctx) -> str:
    me, opp = ctx.me, ctx.opp
    dist = math.hypot(opp[0] - me[0], opp[1] - me[1])
    if ctx.poss_steps >= SAFE_CARRY_STEPS and dist <= ctx.speed * 2 + TACKLE_CONTACT:
        return navigate(ctx, opp, allow_stay=False)  # a tackle needs a non-STAY move
    return defensive_move(ctx, opp, pull=defend_goal_side(ctx), spread=True)


def tactical_action(observation: dict[str, Any], no_kick: bool = False, variant: int = 0,
                    deadline: float | None = None) -> dict[str, Any]:
    ctx = Ctx(observation, deadline)
    ctx.no_kick = no_kick
    ctx.dribble_bias = BIAS_CYCLE[(ctx.conceded + variant) % len(BIAS_CYCLE)]
    if ctx.possession == ctx.player_id:
        return attack_with_ball(ctx)
    if ctx.possession == ctx.opponent_id:
        return {"move": press_carrier(ctx)}
    return {"move": chase_loose_ball(ctx)}


def fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
    ctx = Ctx(observation)
    if ctx.possession == ctx.player_id:
        move = navigate(ctx, ctx.opp_goal, allow_stay=False)
        if ctx.poss_steps >= 5:
            return {"move": move, "kick": {"direction": ctx.attack, "power": 3}}
        return {"move": move}
    target = ctx.opp if ctx.possession == ctx.opponent_id else (ctx.bx, ctx.by)
    return {"move": navigate(ctx, target, allow_stay=False)}


class Policy:

    def __init__(self, *_args, **_kwargs) -> None:
        self.last_sig = None
        self.stuck = 0
        self.no_kick_until = -1
        self.seen: dict = {}
        self.variant = 0
        self.errors = 0

    @classmethod
    def load(cls, path=None) -> "Policy":
        return cls()

    def _stall_guard(self, observation: dict[str, Any]) -> bool:
        state = observation["state"]
        me = state["players"][observation["player_id"]]
        ball = state["ball"]
        iteration = int(state.get("iteration", 0))
        sig = (round(float(me["x"]), 1), round(float(me["y"]), 1), round(float(ball["x"]), 1),
               round(float(ball["y"]), 1), ball.get("possession"))
        holding = ball.get("possession") == observation["player_id"]
        self.stuck = self.stuck + 1 if (holding and sig == self.last_sig) else 0
        self.last_sig = sig
        if self.stuck >= 2:
            self.no_kick_until = iteration + 4
        return iteration <= self.no_kick_until

    def _loop_detect(self, observation: dict[str, Any]) -> None:
        if not F["loop_detect"]:
            return
        state = observation["state"]
        pid, oid = observation["player_id"], observation["opponent_id"]
        me, opp, ball = state["players"][pid], state["players"][oid], state["ball"]
        score = state.get("score") or {}
        sig = (round(float(me["x"]), 1), round(float(me["y"]), 1), round(float(opp["x"]), 1),
               round(float(opp["y"]), 1), round(float(ball["x"]), 1), round(float(ball["y"]), 1),
               ball.get("possession"), int(ball.get("possession_steps", 0)),
               int(score.get(pid, 0)), int(score.get(oid, 0)))
        count = self.seen.get(sig, 0) + 1
        self.seen[sig] = count
        if len(self.seen) > 4000:
            self.seen.clear()
        if count >= 3:
            self.variant += 1
            self.seen.clear()

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            guard = self._stall_guard(observation)
            self._loop_detect(observation)
            return tactical_action(observation, guard, self.variant, started + TIME_BUDGET)
        except Exception as error:  # never freeze: log once in a while and play something sensible
            self.errors += 1
            if self.errors <= 5:
                print(f"policy error: {error!r}", file=sys.stderr, flush=True)
            try:
                return fallback_action(observation)
            except Exception:
                return {"move": "STAY"}
