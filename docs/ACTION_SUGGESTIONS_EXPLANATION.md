# Action Suggestions Module - Complete Explanation

## Overview

The Action Suggestions module (`ai_module/action_suggester.py`) provides real-time recommendations for the player currently in possession of the ball. It evaluates potential actions (passes, dribbles, shots) and ranks them by expected success and value.

---

## 🎯 What It Does

For each frame where a player has the ball (`has_ball == True`), the module:
1. Identifies nearby teammates and opponents
2. Evaluates candidate actions (passes to teammates, dribbles, shots)
3. Scores each action based on multiple factors
4. Returns the top 3 suggestions with detailed features

---

## 📊 How Actions Are Measured

### 1. **Pass Suggestions**

#### Distance to Teammate
- **Measurement**: Euclidean distance from ball owner to potential receiver
- **Formula**: `distance = sqrt((x1-x2)² + (y1-y2)²)`
- **Units**: Meters (if `meters_per_pixel` provided) or pixels
- **Impact**: Shorter passes are generally more successful
- **Threshold**: Only teammates within `search_radius_m` (default: 25m) are considered

#### Receiver Openness Score
- **Measurement**: Number of opponents within `openness_radius_m` (default: 8m) of the potential receiver
- **Method**: Uses KD-tree spatial query for efficient proximity checks
- **Interpretation**: 
  - Lower score = receiver is more "open" (fewer nearby opponents)
  - Higher score = receiver is under pressure
- **Impact**: Open receivers increase pass success probability

#### Line-of-Pass Clearance
- **Measurement**: Probability that an opponent will intercept the pass
- **Method**: 
  - Samples 5 points along the line from owner to receiver
  - Checks for opponents within `openness_radius_m / 2` of each sample point
  - Calculates interception probability using exponential decay: `exp(-distance / intercept_radius)`
- **Formula**: `clearance_score = 1 - max_intercept_probability`
- **Impact**: Clear passing lanes increase success probability

#### Angle to Goal
- **Measurement**: Angle between pass direction and direction to goal
- **Method**: Vector math to compute angle difference
- **Impact**: More direct passes toward goal are preferred
- **Note**: Currently uses simplified goal position (needs actual goal coordinates for accuracy)

#### Expected Threat (xT) Delta
- **Measurement**: Change in position value from pass origin to destination
- **Method**: Zone-based lookup table (simplified model)
- **Zones**: 
  - Attacking third (central): +0.05 xT
  - Attacking third (wide): +0.02 xT
  - Penalty box: +0.10 xT
  - Goal area: +0.20 xT
- **Formula**: `xT_delta = xT_after - xT_before`
- **Impact**: Passes that move the ball into more dangerous areas score higher

#### Pass Success Probability
- **Measurement**: Estimated probability that the pass will be successful
- **Formula**: Logistic model combining multiple factors
  ```
  prob = 1.0
  prob *= (1 - min(1.0, distance_m / 50.0))  # Penalize long passes
  prob *= (0.5 + 0.5 * min(1.0, openness_score / 3.0))  # Reward openness
  prob *= (1 - abs(angle_to_goal_deg) / 90.0)  # Penalize wide angles
  prob = max(0.05, min(0.95, prob))  # Clamp between 5% and 95%
  ```
- **Impact**: Primary factor in pass scoring

#### Final Pass Score
- **Formula**: 
  ```
  pass_score = pass_prob + (xT_weight * xT_delta) + (clearance_score * 0.5)
  ```
- **Components**:
  - `pass_prob`: Base success probability (0-1)
  - `xT_delta * xT_weight`: Position value gain (default weight: 0.35)
  - `clearance_score * 0.5`: Passing lane quality (weight: 0.5)
- **Threshold**: Only passes with `pass_score > min_pass_prob` (default: 0.25) are suggested

### 2. **Dribble Suggestions**

#### Space Ahead
- **Measurement**: Clear space in front of the player
- **Method**: Checks for opponents in forward direction
- **Threshold**: Requires at least `dribble_space_m` (default: 12m) of clear space
- **Impact**: More space = higher dribble score

#### Pressure Nearby
- **Measurement**: Number of opponents within `dribble_pressure_radius_m` (default: 3m)
- **Method**: KD-tree query around player position
- **Impact**: Low pressure = better dribble opportunity
- **Score**: `dribble_score = 0.7` if conditions met (heuristic)

### 3. **Shot Suggestions**

#### Distance to Goal
- **Measurement**: Euclidean distance from player to goal
- **Threshold**: Must be within `shot_zone_m` (default: 20m)
- **Impact**: Closer to goal = better shot opportunity

#### Pressure
- **Measurement**: Number of opponents nearby
- **Threshold**: Requires `pressure_nearby < 2`
- **Impact**: Low pressure = better shot opportunity
- **Score**: `shot_score = 0.8` if conditions met (heuristic)

---

## 🔧 Tunable Parameters

All parameters can be adjusted via command-line arguments:

### Search & Proximity
- `--action_search_radius_m` (default: 25.0)
  - Maximum distance to consider teammates/opponents
  - Increase to find more distant options
  - Decrease for more conservative suggestions

- `--action_openness_radius_m` (default: 8.0)
  - Radius to check for opponents around receiver
  - Increase to be more strict about "openness"
  - Decrease to allow tighter spaces

### Scoring Thresholds
- `--action_min_pass_prob` (default: 0.25)
  - Minimum pass score to be suggested
  - Increase to show only high-confidence passes
  - Decrease to show more options

- `--action_xt_weight` (default: 0.35)
  - Weight for expected threat in pass scoring
  - Increase to prioritize dangerous passes
  - Decrease to focus on safe passes

- `--action_intercept_threshold` (default: 0.6)
  - Probability threshold for interception detection
  - Increase to be more sensitive to interceptors
  - Decrease to allow riskier passes

### Action-Specific
- `shot_zone_m` (default: 20.0) - Maximum distance for shot suggestions
- `dribble_space_m` (default: 12.0) - Minimum clear space for dribble
- `dribble_pressure_radius_m` (default: 3.0) - Radius to check for pressure

---

## 📤 Output Format

### Action Suggestions CSV
Each row contains:
- `frame`: Frame number
- `owner_id`: Player ID with the ball
- `suggestion_type`: "pass", "dribble", or "shot"
- `target_id`: Target player ID (for passes)
- `score`: Action score (0-1)
- `features`: JSON string with detailed metrics

### Features JSON Structure
```json
{
  "distance_m": 15.3,
  "openness_score": 1.2,
  "clearance_score": 0.85,
  "xt_delta": 0.05,
  "pass_prob": 0.72,
  "pressure_nearby": 2,
  "space_ahead_m": 8.5
}
```

### Visualization Overlay
- **Pass suggestions**: Orange arrows from owner to target
- **Dribble suggestions**: Yellow text label "DRIBBLE [score]"
- **Shot suggestions**: Red text label "SHOT [score]"
- Only top 2 suggestions are shown to avoid clutter

---

## 🎓 Understanding the Scores

### Pass Scores (0-1 scale)
- **0.7-1.0**: Excellent pass opportunity (high success probability, good position)
- **0.5-0.7**: Good pass (moderate risk, decent reward)
- **0.25-0.5**: Risky pass (low success probability or poor position)
- **< 0.25**: Not suggested (below threshold)

### Factors That Increase Pass Score
✅ Short distance to receiver
✅ Receiver is open (few nearby opponents)
✅ Clear passing lane (no interceptors)
✅ Pass moves ball toward goal
✅ Receiver in dangerous position (high xT)

### Factors That Decrease Pass Score
❌ Long distance
❌ Receiver under pressure
❌ Opponents on passing path
❌ Wide angle away from goal
❌ Receiver in safe/defensive position

---

## 🔍 Debugging

### Enable Debug Output
```bash
python main.py --enable_action_suggestions --action_debug_frame 100 --action_debug_owner 5
```

This will print detailed candidate features for:
- Frame 100
- Player ID 5 (if they have the ball)

### Debug Output Format
```
--- Debug Suggestions for Frame 100 (Owner: 5) ---
  Type: pass, Target: 12, Score: 0.78, Features: {'distance_m': 12.5, 'openness_score': 0.8, ...}
  Type: pass, Target: 8, Score: 0.65, Features: {'distance_m': 18.2, 'openness_score': 1.5, ...}
  Type: dribble, Target: N/A, Score: 0.70, Features: {'space_ahead_m': 15.0, 'pressure_nearby': 1}
--------------------------------------------------
```

---

## 📊 Example Interpretation

### Scenario: Player 5 has the ball

**Suggested Pass to Player 12:**
- Score: 0.78
- Distance: 12.5m
- Openness: 0.8 (very open)
- Clearance: 0.85 (clear lane)
- xT Delta: +0.05 (moving into attacking zone)
- **Interpretation**: Excellent short pass to an open teammate in a dangerous position

**Suggested Pass to Player 8:**
- Score: 0.65
- Distance: 18.2m
- Openness: 1.5 (moderate pressure)
- Clearance: 0.70 (some risk)
- xT Delta: +0.02 (slight improvement)
- **Interpretation**: Good option but riskier due to distance and pressure

**Suggested Dribble:**
- Score: 0.70
- Space ahead: 15.0m
- Pressure: 1 opponent nearby
- **Interpretation**: Decent dribble opportunity with space to advance

---

## 🚀 Usage

### Basic Usage
```bash
python main.py --video input_videos/match.mp4 --enable_action_suggestions
```

### Tuned for Aggressive Suggestions
```bash
python main.py --video input_videos/match.mp4 --enable_action_suggestions \
  --action_min_pass_prob 0.15 \
  --action_xt_weight 0.5 \
  --action_search_radius_m 30.0
```

### Tuned for Conservative Suggestions
```bash
python main.py --video input_videos/match.mp4 --enable_action_suggestions \
  --action_min_pass_prob 0.4 \
  --action_intercept_threshold 0.8 \
  --action_openness_radius_m 10.0
```

---

## ⚠️ Limitations & Future Improvements

### Current Limitations
1. **Simplified xT Model**: Uses zone-based lookup instead of learned model
2. **Goal Position**: Assumes goal at fixed position (needs actual goal detection)
3. **Heuristic Scoring**: Dribble and shot scores are fixed heuristics
4. **No Player Roles**: Doesn't consider player positions/roles
5. **No Context**: Doesn't account for game state (score, time, etc.)

### Future Enhancements
- **Learned Models**: Train ML models on real match data for pass success probability
- **Advanced xT**: Use state-of-the-art expected threat models
- **Player Roles**: Incorporate positional data (defender, midfielder, forward)
- **Game Context**: Consider score, time remaining, tactical situation
- **Real-Time Optimization**: Optimize for live analysis with lower latency

---

## 📝 Notes

- All measurements use world coordinates (meters) when `meters_per_pixel` is provided
- Spatial queries use KD-trees for efficient nearest-neighbor searches
- Only top 3 suggestions are returned to avoid information overload
- Suggestions are frame-by-frame and don't consider multi-step sequences
- Action suggestions are recommendations, not guarantees of success

