"""
Advanced Seek Agent implementation with optimized strategies
- A* with predictive movement + Cut-off blocking
"""

import sys
from pathlib import Path
from collections import deque
from heapq import heappush, heappop
import numpy as np
import time


src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


class BaseAgentLogic:
    """Helper functions for pathfinding."""

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Get all valid neighboring positions and their moves."""
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))

        return neighbors

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def a_star(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """A* pathfinding algorithm."""
        heap = [(0, start, [])]
        visited = {start: 0}

        while heap:
            f_cost, current_pos, path = heappop(heap)

            if current_pos == goal:
                return path

            g_cost = len(path)

            for next_pos, move in self._get_neighbors(current_pos, map_state):
                new_g = g_cost + 1

                if next_pos not in visited or new_g < visited[next_pos]:
                    visited[next_pos] = new_g
                    h_cost = self._manhattan_distance(next_pos, goal)
                    f_cost = new_g + h_cost
                    heappush(heap, (f_cost, next_pos, path + [move]))

        return []


class PacmanAgent(BasePacmanAgent, BaseAgentLogic):
    """
    Advanced Seek Agent with:
    - A* for optimal pathfinding
    - Predictive movement (anticipate enemy's next position)
    - Cut-off strategy (block escape routes)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Advanced Seek Agent"
        self.current_path = []
        self.last_enemy_pos = None
        self.enemy_history = deque(maxlen=5)

    def _predict_enemy_move(
        self, enemy_pos: tuple, my_pos: tuple, map_state: np.ndarray
    ) -> tuple:
        """Predict where enemy will move next (away from us)."""
        if len(self.enemy_history) < 2:
            return enemy_pos

        prev_pos = self.enemy_history[-2]
        delta_row = enemy_pos[0] - prev_pos[0]
        delta_col = enemy_pos[1] - prev_pos[1]

        predicted = (enemy_pos[0] + delta_row, enemy_pos[1] + delta_col)

        if self._is_valid_position(predicted, map_state):
            return predicted

        best_pos = enemy_pos
        max_dist = -1

        for next_pos, _ in self._get_neighbors(enemy_pos, map_state):
            dist = self._manhattan_distance(next_pos, my_pos)
            if dist > max_dist:
                max_dist = dist
                best_pos = next_pos

        return best_pos

    def _find_cutoff_target(
        self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray
    ) -> tuple:
        """Find strategic position to cut off enemy's escape routes."""

        enemy_reachable = set()
        queue = deque([(enemy_pos, 0)])
        visited = {enemy_pos}

        while queue:
            pos, depth = queue.popleft()
            if depth > 3:
                continue
            enemy_reachable.add(pos)

            for next_pos, _ in self._get_neighbors(pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, depth + 1))

        best_target = enemy_pos
        min_escape_options = float("inf")

        for candidate in enemy_reachable:

            escape_count = 0
            for enemy_next in enemy_reachable:
                if self._manhattan_distance(
                    candidate, enemy_next
                ) > self._manhattan_distance(candidate, enemy_pos):
                    escape_count += 1

            if escape_count < min_escape_options:
                min_escape_options = escape_count
                best_target = candidate

        return best_target

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:

        start_time = time.time()

        self.enemy_history.append(enemy_position)

        target = enemy_position

        if self._manhattan_distance(my_position, enemy_position) <= 5:
            cutoff_target = self._find_cutoff_target(
                my_position, enemy_position, map_state
            )
            if cutoff_target != enemy_position:
                target = cutoff_target

        if not self.current_path or self.last_enemy_pos != enemy_position:
            self.current_path = self.a_star(my_position, target, map_state)
            self.last_enemy_pos = enemy_position

        if time.time() - start_time > 0.9:

            if self.current_path:
                return self.current_path[0]
            return Move.STAY

        if self.current_path:
            next_move = self.current_path.pop(0)
            return next_move

        return Move.STAY
