"""
Agent interface definition for students to implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from environment import Move


class AgentInterface(ABC):
    """
    Base interface that all student agents must implement.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the agent.
        Students can use this to set up any data structures they need.
        
        Args:
            **kwargs: Optional arguments for agent configuration
        """
        pass
    
    @abstractmethod
    def step(self, map_state: np.ndarray, 
             my_position: Tuple[int, int], 
             enemy_position: Tuple[int, int],
             step_number: int) -> Move:
        """
        Decide the next move based on the current environment.
        
        Args:
            map_state: 2D numpy array where 1 = wall, 0 = empty space
            my_position: Current position as (row, col)
            enemy_position: Enemy's current position as (row, col)
            step_number: Current step number in the game
            
        Returns:
            Move enum value (UP, DOWN, LEFT, RIGHT, or STAY)
        """
        pass


class PacmanAgent(AgentInterface):
    """
    Interface for Pacman agents (seekers).
    Goal: Catch the ghost.
    """
    pass


class GhostAgent(AgentInterface):
    """
    Interface for Ghost agents (hiders).
    Goal: Avoid being caught by Pacman.
    """
    pass
