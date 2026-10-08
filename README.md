# Conv-Cup '26 Submission: botsync

## 1. Executable agent
Launch command (from `submission.json`), run from the archive root:

```
python -m team_bot.bot --model team_bot/models/policy.json
```

The agent implements the prescribed interface: each iteration it receives one observation and returns one action.
It is deterministic (no randomness), so the same seed, start state and opponent actions give the same actions.

## 2. Source code
- `team_bot/policy.py`: the agent (all decision logic)
- `team_bot/bot.py`: platform entry point, taken unmodified from the official starter kit
- `team_bot/__init__.py`: package marker

## 3. Documentation

### Use of the state / action interface
**Reads from the observation:** `player_id`, `opponent_id`, `attack_direction`; in `state`: field size and goal width,
player radius and speed, both players' positions, the ball (position, velocity, status, possession,
`possession_steps`, `loose_ball_steps`, remaining kick distance), obstacles, score, iteration and maximum iterations.

**Returns:** `{"move": <8 directions or STAY>}`, optionally with `"kick": {"direction": <direction>, "power": 1|2|3}`.
Only the agent's own player is controlled. The agent never modifies game state or the simulator.

### Approach
A rule-based agent that uses simulated lookahead of the game's own physics.
- **Ball model:** kicks are simulated with wall and obstacle bounces, goal mouth, and the three kick distances;
  candidates are pruned with a coarse trace and the best are re-checked at the engine's fine resolution.
  Interception chances, own-goal risk and bounce-back onto the kicker are scored.
- **With the ball:** chooses a (move, kick) pair jointly, evaluating each kick from the position the player will
  occupy after moving. Otherwise it carries the ball toward the nearest cell with a scoring lane, and it kicks before
  a tackle becomes possible or before the forced release at 10 possession steps.
- **Opponent has the ball:** tackles when legal; otherwise moves to minimise the opponent's best shot,
  including shots taken after the opponent's next step.
- **Loose ball:** obstacle-aware race to the ball, interception-point chasing for moving balls, and a standoff
  breaker using `loose_ball_steps`.
- **Navigation:** grid distance fields (Dijkstra), cached per obstacle layout, so paths avoid obstacles.
- **Match state:** playing style adapts to score, time remaining and goals remaining; a repeated-state detector
  changes tactic if the game falls into a loop.

### Reliability
- Decisions are time-budgeted at 0.8 s (limit: 2 s); typical decisions take a few to tens of milliseconds.
- Any internal error falls back to a simple safe action, so the agent always returns a valid response.

## 4. Dependencies and execution instructions
- Python 3.11 or newer.
- Standard library only. No third-party packages (`requirements.txt` lists none).
- No external APIs, no network access, and no file writes at run time.
- Verified with the official `validate_submission.py` (0 action errors) and `check_submission.py` (pass).