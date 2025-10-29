"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- You MUST return a Move enum value from step()
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""
import time
from collections import deque
import math
import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        # - self.name = "Your Agent Name"
        pass

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here

        # Example: Simple greedy approach (replace with your algorithm)
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]

        # Try to move towards ghost
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT

        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move

        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move

        return Move.STAY

    # Helper methods (you can add more)

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught

    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        pass

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col)
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        
        # Bước 1: Tìm top 5 vị trí xa Pacman nhất bằng BFS
        safe_positions = self._find_safe_positions(
            map_state, my_position, enemy_position, top_k=5
        )
        
        # Nếu không tìm được vị trí nào → dùng fallback (chạy ngược hướng)
        if not safe_positions:
            return self._greedy_escape(my_position, enemy_position, map_state)
        
        # Bước 2: Trong 5 vị trí xa nhất, chọn vị trí có NHIỀU LỐI THOÁT NHẤT
        # max() với key=lambda sẽ chọn vị trí có _count_escape_routes() cao nhất
        best_position = max(
            safe_positions,
            key=lambda pos: self._count_escape_routes(pos, map_state)
        )
        
        # Bước 3: Tìm bước đi ĐẦU TIÊN để đến vị trí best_position
        next_move = self._get_next_move(my_position, best_position, map_state)
        
        # Trả về bước đi (hoặc STAY nếu không tìm được)
        return next_move if next_move else Move.STAY

    # Helper methods (you can add more)

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0
    
    def _find_safe_positions(self, map_state, my_pos, enemy_pos, top_k=5):
        """ 
        Tìm top_k vị trí xa Pacman nhất mà Ghost có thể đến được

        # Sử dụng BFS để duyệt toàn bộ map và tìm các vị trí xa nhất

        # Returns về list các vị trí [(row, col), ...] xa Pacman nhất
        """
        
        # Bước 1: Chuẩn bị BFS

        visited = set()                # Lưu các ô đã thăm
        queue = deque([(my_pos, 0)])         # Queue: (vị trí, khoảng cách từ ghost)
        visited.add(my_pos)
        positions_with_distance = []         # Lưu: (vị trí, khoảng cách đến Pacman)
        
        # Bước 2: BFS duyệt toàn bộ map
        while queue:
            current_pos, dist_from_start = queue.popleft()
            
            # Tính khoảng cách Manhattan từ current_pos đến Pacman
            # Manhattan distance = |row1 - row2| + |col1 - col2|
            dist_to_enemy = abs(current_pos[0] - enemy_pos[0]) + abs(current_pos[1] - enemy_pos[1])
            
            # Lưu vị trí này cùng khoảng cách đến Pacman
            positions_with_distance.append((current_pos, dist_to_enemy))
            
            # Duyệt 4 hướng (UP, DOWN, LEFT, RIGHT)
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value  # Lấy delta_row, delta_col
                new_pos = (current_pos[0] + dr, current_pos[1] + dc)
                
                # Kiểm tra new_pos có hợp lệ không?
                #           Chưa được thăm (not in visited)
                #           Hợp lệ (_is_valid_position)
                if new_pos not in visited and self._is_valid_position(new_pos, map_state):
                    visited.add(new_pos)
                    queue.append((new_pos, dist_from_start + 1))
        
        # Bước 3: Sắp xếp theo khoảng cách đến Pacman (giảm dần)
        # Vị trí xa nhất sẽ ở đầu list
        positions_with_distance.sort(key=lambda x: x[1], reverse=True)
        
        # Bước 4: Trả về top_k vị trí (chỉ lấy vị trí, bỏ khoảng cách)
        return [pos for pos, dist in positions_with_distance[:top_k]]
    
    def _count_escape_routes(self, pos, map_state):
        """
        Đếm số lối thoát từ một vị trí
            
        Returns:
            int: Số lượng ô trống xung quanh (0-4)
        """
        count = 0
        
        # Duyệt 4 hướng và đếm số ô hợp lệ
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            neighbor_pos = (pos[0] + dr, pos[1] + dc)
            
            if self._is_valid_position(neighbor_pos, map_state):
                count += 1
        
        return count
    
    def _get_next_move(self, start, goal, map_state):
        """
        Tìm bước đi đầu tiên để đi từ start đến goal.
        
        Sử dụng BFS để tìm đường ngắn nhất, rồi trả về bước đầu tiên.
        
        Returns:
            Move: Bước đi đầu tiên (UP/DOWN/LEFT/RIGHT/STAY)
        """
        from collections import deque
        
        if start == goal:
            return Move.STAY
        
        # BFS với lưu path (đường đi)
        visited = {start}
        queue = deque([(start, [])])  # (vị trí, [list các move đã đi])
        
        while queue:
            pos, path = queue.popleft()
            
            # Thử 4 hướng
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (pos[0] + dr, pos[1] + dc)
                
                # Nếu đến đích → trả về bước ĐẦU TIÊN trong path
                if new_pos == goal:
                    if path:  # Nếu đã có path trước đó
                        return path[0]
                    else:     # Nếu goal ngay cạnh start
                        return move
                
                # Nếu chưa thăm và hợp lệ → thêm vào queue
                if new_pos not in visited and self._is_valid_position(new_pos, map_state):
                    visited.add(new_pos)
                    new_path = path + [move]  # Thêm move vào path
                    queue.append((new_pos, new_path))
        
        # Không tìm được đường → đứng yên
        return Move.STAY
    
    def _greedy_escape(self, my_pos, enemy_pos, map_state):
        """
        Fallback: Chạy ngược hướng Pacman đơn giản (dùng khi không tìm được vị trí tốt).
        
        Args:
            my_pos: Vị trí hiện tại của Ghost
            enemy_pos: Vị trí của Pacman
            map_state: Bản đồ
            
        Returns:
            Move: Bước đi tránh Pacman
        """
        row_diff = my_pos[0] - enemy_pos[0]
        col_diff = my_pos[1] - enemy_pos[1]

        # Chạy ngược hướng Pacman
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT

        # Kiểm tra move có hợp lệ không
        if self._is_valid_move(my_pos, move, map_state):
            return move

        # Nếu không hợp lệ, thử các hướng khác
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move, map_state):
                return move

        return Move.STAY
        
        