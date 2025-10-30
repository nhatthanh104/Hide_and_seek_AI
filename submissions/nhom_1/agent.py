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
        # Path caching để tránh tính toán lại mỗi step
        self.current_path = []
        self.last_enemy_pos = None
        
        print("🔵 Pacman Agent: A* search with path replanning activated!")

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        """
        🎯 A* SEARCH với PATH REPLANNING
        
        Chiến lược:
        1. Chỉ replan khi cần thiết (Ghost di chuyển xa hoặc chưa có path)
        2. Follow cached path để tiết kiệm tính toán
        3. A* đảm bảo luôn tìm được đường ngắn nhất
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        
        # Điều kiện replan:
        # 1. Chưa có path
        # 2. Lần đầu chạy (chưa biết vị trí Ghost trước đó)
        # 3. Ghost di chuyển đáng kể (> 2 cells) - có thể đổi hướng
        should_replan = (
            not self.current_path or 
            self.last_enemy_pos is None or
            self._manhattan_distance(enemy_position, self.last_enemy_pos) > 2
        )
        
        if should_replan:
            # Tính path mới bằng A*
            self.current_path = self._astar(my_position, enemy_position, map_state)
            self.last_enemy_pos = enemy_position
        
        # Follow cached path
        if self.current_path:
            next_move = self.current_path.pop(0)
            return next_move
        
        # Fallback: Không tìm được đường (không thể xảy ra trong map liên thông)
        # Thử di chuyển gần Ghost nhất
        best_move = Move.STAY
        best_distance = float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            new_pos = (my_position[0] + dr, my_position[1] + dc)
            
            if self._is_valid_position(new_pos, map_state):
                distance = self._manhattan_distance(new_pos, enemy_position)
                if distance < best_distance:
                    best_distance = distance
                    best_move = move
        
        return best_move

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
    
    def _manhattan_distance(self, pos1, pos2):
        """Tính khoảng cách Manhattan giữa 2 vị trí."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _astar(self, start, goal, map_state):
        """
        A* Search: Tìm đường ngắn nhất từ start đến goal.
        
        A* = Best-First Search + Heuristic
        f(n) = g(n) + h(n)
        - g(n): Cost từ start đến n (số bước đã đi)
        - h(n): Heuristic ước lượng từ n đến goal (Manhattan distance)
        
        Args:
            start: Vị trí bắt đầu (row, col)
            goal: Vị trí đích (row, col)
            map_state: Bản đồ
            
        Returns:
            List[Move]: Danh sách các bước đi, hoặc [] nếu không tìm được
        """
        from heapq import heappush, heappop
        
        # Priority queue: (f_cost, g_cost, position, path)
        # g_cost để break tie khi f_cost bằng nhau
        frontier = [(0, 0, start, [])]
        visited = set()
        
        while frontier:
            f_cost, g_cost, current_pos, path = heappop(frontier)
            
            # Đến đích!
            if current_pos == goal:
                return path
            
            # Đã thăm rồi → bỏ qua
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            
            # Explore 4 hướng
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (current_pos[0] + dr, current_pos[1] + dc)
                
                # Kiểm tra valid và chưa thăm
                if next_pos not in visited and self._is_valid_position(next_pos, map_state):
                    new_path = path + [move]
                    new_g_cost = len(new_path)  # Cost từ start (số bước)
                    h_cost = self._manhattan_distance(next_pos, goal)  # Heuristic
                    new_f_cost = new_g_cost + h_cost  # Total cost
                    
                    heappush(frontier, (new_f_cost, new_g_cost, next_pos, new_path))
        
        # Không tìm được đường
        return []


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
        Kết hợp Minimax (gần) và Predictive BFS (xa)

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col)
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # Tính khoảng cách đến Pacman
        distance_to_pacman = self._calculate_manhattan_distance(my_position, enemy_position)
        
        # Vùng nguy hiểm : Nếu gần Pacman (<=6) → dùng Minimax để chọn nước đi tốt nhất
        if distance_to_pacman <= 6:
            depth = 3  # Gần - depth trung bình
            if distance_to_pacman <= 3:
                depth = 4  # Rất gần - depth cao
                
            #Gọi minimax với depth=3
            _, best_move = self._minimax(
                my_position, enemy_position, 
                depth=depth,  # Nhìn trước 3 bước
                is_ghost_turn=True, 
                map_state=map_state
            )
            
            # Nếu minimax trả về move hợp lệ -> dùng
            if best_move != Move.STAY:
                return best_move
            
            # Fallback nếu minimax thất bại
            return self._greedy_escape(my_position, enemy_position, map_state)
        
        # Vùng an toàn : Pacman XA (> 6 cells) → Dùng PREDICTIVE BFS
        else:
            # Bước 1 - Dự đoán Pacman sẽ ở đâu 
            predicted_enemy_pos = self._predict_enemy_position(
                enemy_position, my_position, map_state
            )
            
            # Bước 2 - Tìm vị trí xa predicted position
            safe_positions = self._find_safe_positions(
                map_state, my_position, predicted_enemy_pos, top_k=5
            )
            
            if not safe_positions:
                return self._greedy_escape(my_position, enemy_position, map_state)
            
            # Bước 3 - Chọn vị trí có nhiều lối thoát nhất  
            best_position = max(
                safe_positions,
                key=lambda pos: self._count_escape_routes(pos, map_state)
            )
            
            # Bước 4 - Di chuyển đến vị trí đó
            next_move = self._get_next_move(my_position, best_position, map_state)
            
            return next_move if next_move else Move.STAY
        
        
        # # Bước 1: Tìm top 5 vị trí xa Pacman nhất bằng BFS
        # safe_positions = self._find_safe_positions(
        #     map_state, my_position, enemy_position, top_k=5
        # )

        # # Nếu không tìm được vị trí nào → dùng fallback (chạy ngược hướng)
        # if not safe_positions:
        #     return self._greedy_escape(my_position, enemy_position, map_state)

        # # Bước 2: Trong 5 vị trí xa nhất, chọn vị trí có NHIỀU LỐI THOÁT NHẤT
        # # max() với key=lambda sẽ chọn vị trí có _count_escape_routes() cao nhất
        # best_position = max(
        #     safe_positions,
        #     key=lambda pos: self._count_escape_routes(pos, map_state)
        # )

        # # Bước 3: Tìm bước đi ĐẦU TIÊN để đến vị trí best_position
        # next_move = self._get_next_move(my_position, best_position, map_state)

        # # Trả về bước đi (hoặc STAY nếu không tìm được)
        # return next_move if next_move else Move.STAY

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

    def _calculate_manhattan_distance(self, pos1, pos2):
        # Tính khoảng cách Manhattan giữa hai vị trí
        # Manhattan distance = |row1 - row2| + |col1 - col2|
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
            dist_to_enemy = self._calculate_manhattan_distance(current_pos, enemy_pos)

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
    
    def _predict_enemy_position(self, enemy_pos, my_pos, map_state):
        """
        Dự đoán Pacman sẽ di chuyển đến đâu 
    
        Giả định: Pacman sẽ chọn bước đi gần Ghost nhất (greedy)
        
        Returns:
            tuple: Vị trí dự đoán của Pacman ở bước tiếp theo
        """
        best_move = Move.STAY
        best_distance = float('inf')  # Pacman muốn minimize khoảng cách
        
        #Duyệt 4 hướng Pacman có thể đi
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            new_pos = (enemy_pos[0] + dr, enemy_pos[1] + dc)
            
            #Kiểm tra valid
            if not self._is_valid_position(new_pos, map_state):
                continue
            
            #Tính khoảng cách từ new_pos đến Ghost (my_pos)
            distance = self._calculate_manhattan_distance(new_pos, my_pos)
            
            #Pacman chọn move làm distance nhỏ nhất
            if distance < best_distance:
                best_distance = distance
                best_move = move
        
        #Apply move để có predicted position
        if best_move != Move.STAY:
            dr, dc = best_move.value
            return (enemy_pos[0] + dr, enemy_pos[1] + dc)
        
        return enemy_pos  
        
    def _minimax(self, my_pos, enemy_pos, depth, is_ghost_turn, map_state):
        """
        Minimax algorithm cho Ghost 
        
        Ghost = Maximizing player (muốn xa Pacman)
        Pacman = Minimizing player (muốn gần Ghost)
        
        Args:
            depth: Số bước nhìn trước (3-4 là tốt)
            is_ghost_turn: True = lượt Ghost, False = lượt Pacman
            
        Returns:
            (score, best_move): Score là khoảng cách, move là nước đi tốt nhất
        """
        
        # Hết depth hoặc bị bắt
        if depth == 0 or my_pos == enemy_pos:
            #Return khoảng cách Manhattan
            distance = self._calculate_manhattan_distance(my_pos, enemy_pos)
            return distance, Move.STAY
        
        if is_ghost_turn:  # Lượt Ghost - MAXIMIZE distance
            best_score = -float('inf')
            best_move = Move.STAY
            
            #Thử 4 hướng Ghost có thể đi
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (my_pos[0] + dr, my_pos[1] + dc)
                
                #Kiểm tra valid
                if not self._is_valid_position(new_pos, map_state):
                    continue
                
                #Gọi đệ quy - lượt Pacman (is_ghost_turn=False)
                score, _ = self._minimax(new_pos, enemy_pos, depth - 1, False, map_state)
                
                #Cập nhật best nếu score CAO hơn (maximize)
                if score > best_score:
                    best_score = score
                    best_move = move
            
            return best_score, best_move
        
        else:  #Lượt Pacman - MINIMIZE distance
            best_score = float('inf')
            best_move = Move.STAY
            
            #Thử 4 hướng Pacman có thể đi
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (enemy_pos[0] + dr, enemy_pos[1] + dc)
                
                #Kiểm tra valid
                if not self._is_valid_position(new_pos, map_state):
                    continue
                
                #Gọi đệ quy - lượt Ghost (is_ghost_turn=True)
                score, _ = self._minimax(my_pos, new_pos, depth - 1, True, map_state)
                
                #Cập nhật best nếu score THẤP hơn (minimize)
                if score < best_score:
                    best_score = score
                    best_move = move
            
            return best_score, best_move