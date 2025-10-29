"""
Environment module for the Pacman vs Ghost arena.
Defines the game map, positions, and rules.
"""

import numpy as np
from typing import Tuple, List, Optional
from enum import Enum


class CellType(Enum):
    """Types of cells in the game map."""
    EMPTY = 0
    WALL = 1
    PACMAN = 2
    GHOST = 3


class Move(Enum):
    """Valid moves for agents."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    STAY = (0, 0)


class Environment:
    """
    Game environment that manages the map and agent positions.
    """
    
    def __init__(self, map_layout: Optional[np.ndarray] = None, max_steps: int = 200):
        """
        Initialize the environment.
        
        Args:
            map_layout: 2D numpy array where 1 = wall, 0 = empty
            max_steps: Maximum number of steps before game ends in a draw
        """
        if map_layout is None:
            # Default classic Pacman-style map
            self.map = self._create_default_map()
        else:
            self.map = map_layout.copy()
        
        self.height, self.width = self.map.shape
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize positions
        self.pacman_pos = None
        self.ghost_pos = None
        self.reset()
    
    def _create_default_map(self) -> np.ndarray:
        """
        Create a default Pacman-style map.
        1 = wall, 0 = empty space
        """
        layout = [
            "#####################",
            "#.........#.........#",
            "#.###.###.#.###.###.#",
            "#...................#",
            "#.###.#.#####.#.###.#",
            "#.....#...#...#.....#",
            "#####.### # ###.#####",
            "    #.#       #.#    ",
            "#####.# ##-## #.#####",
            "     .  #   #  .     ",
            "#####.# ##### #.#####",
            "    #.#       #.#    ",
            "#####.# ##### #.#####",
            "#.........#.........#",
            "#.###.###.#.###.###.#",
            "#...#.....P.....#...#",
            "###.#.#.#####.#.#.###",
            "#.....#...#...#.....#",
            "#.#######.#.#######.#",
            "#...................#",
            "#####################"
        ]
        
        map_array = np.zeros((len(layout), len(layout[0])), dtype=int)
        for i, row in enumerate(layout):
            for j, cell in enumerate(row):
                if cell == '#':
                    map_array[i, j] = 1
                elif cell == '-':
                    map_array[i, j] = 1
        
        return map_array
    
    def reset(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Reset the environment to initial state.
        
        Returns:
            Tuple of (map, pacman_position, ghost_position)
        """
        self.current_step = 0
        
        # Find valid starting positions (empty cells)
        empty_cells = np.argwhere(self.map == 0)
        
        # Set Pacman at bottom area
        bottom_cells = empty_cells[empty_cells[:, 0] > self.height * 0.6]
        if len(bottom_cells) > 0:
            pacman_idx = np.random.choice(len(bottom_cells))
            self.pacman_pos = tuple(bottom_cells[pacman_idx])
        else:
            self.pacman_pos = tuple(empty_cells[0])
        
        # Set Ghost at top area
        top_cells = empty_cells[empty_cells[:, 0] < self.height * 0.4]
        if len(top_cells) > 0:
            ghost_idx = np.random.choice(len(top_cells))
            self.ghost_pos = tuple(top_cells[ghost_idx])
        else:
            self.ghost_pos = tuple(empty_cells[-1])
        
        return self.get_state()
    
    def get_state(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Get the current state of the environment.
        
        Returns:
            Tuple of (map, pacman_position, ghost_position)
        """
        return self.map.copy(), self.pacman_pos, self.ghost_pos
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (within bounds and not a wall).
        
        Args:
            pos: (row, col) position
            
        Returns:
            True if position is valid, False otherwise
        """
        row, col = pos
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return self.map[row, col] == 0
    
    def apply_move(self, current_pos: Tuple[int, int], move: Move) -> Tuple[int, int]:
        """
        Apply a move to a position.
        
        Args:
            current_pos: Current (row, col) position
            move: Move to apply
            
        Returns:
            New position after move (stays same if invalid)
        """
        delta_row, delta_col = move.value
        new_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
        
        if self.is_valid_position(new_pos):
            return new_pos
        return current_pos
    
    def step(self, pacman_move: Move, ghost_move: Move) -> Tuple[bool, str, Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
        """
        Execute one step of the game.
        
        Args:
            pacman_move: Move chosen by Pacman agent
            ghost_move: Move chosen by Ghost agent
            
        Returns:
            Tuple of (game_over, result, new_state)
            - game_over: True if game has ended
            - result: 'pacman_wins', 'ghost_wins', or 'draw'
            - new_state: New state of the environment
        """
        self.current_step += 1
        
        # Apply moves
        new_pacman_pos = self.apply_move(self.pacman_pos, pacman_move)
        new_ghost_pos = self.apply_move(self.ghost_pos, ghost_move)
        
        # Update positions
        self.pacman_pos = new_pacman_pos
        self.ghost_pos = new_ghost_pos
        
        # Check win conditions
        # Pacman catches Ghost
        if self.pacman_pos == self.ghost_pos:
            return True, 'pacman_wins', self.get_state()
        
        # Ghost wins if Pacman fails to catch within the allotted steps
        if self.current_step >= self.max_steps:
            return True, 'ghost_wins', self.get_state()
        
        return False, '', self.get_state()
    
    def get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self) -> str:
        """
        Render the current state as a string.
        
        Returns:
            String representation of the map with agents
        """
        display = self.map.copy().astype(str)
        display[display == '0'] = '.'
        display[display == '1'] = '#'
        
        # Mark agent positions
        if self.pacman_pos:
            display[self.pacman_pos] = 'P'
        if self.ghost_pos:
            display[self.ghost_pos] = 'G'
        
        # If they're at the same position, show collision
        if self.pacman_pos == self.ghost_pos:
            display[self.pacman_pos] = 'X'
        
        rows = [''.join(row) for row in display]
        return '\n'.join(rows)
