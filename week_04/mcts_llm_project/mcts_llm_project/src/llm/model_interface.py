"""
LLM Model Interface Implementation
Author: Mehar
Course: NEU Self-Improving AI
Date: 2025

Provides interface for integrating Large Language Models with MCTS.
Implements the approach from "Language Agent Tree Search" paper.
"""

import json
import time
import logging
from typing import Any, List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.95
    api_key: Optional[str] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0


class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def evaluate(self, state: Any, action: Any = None) -> float:
        """Evaluate state value for MCTS."""
        pass
    
    @abstractmethod
    def get_actions(self, state: Any) -> List[Any]:
        """Generate possible actions from state."""
        pass


class LanguageModelInterface(BaseLLM):
    """
    Interface for integrating LLMs with MCTS.
    
    This implementation follows the LATS paper approach:
    - Using LLM for action generation
    - LLM-based value estimation
    - Reflection and self-evaluation mechanisms
    """
    
    def __init__(self, config: LLMConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize LLM interface.
        
        Args:
            config: LLM configuration
            logger: Optional logger
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        self.generation_cache = {}
        self.evaluation_cache = {}
        
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the language model.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        cache_key = self._create_cache_key(prompt, kwargs)
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
        
        response = self._mock_llm_generate(prompt, **kwargs)
        self.generation_cache[cache_key] = response
        
        return response
    
    def evaluate(self, state: Any, action: Any = None) -> float:
        """
        Evaluate state value using LLM reasoning.
        
        Args:
            state: Current state to evaluate
            action: Optional action taken to reach this state
            
        Returns:
            Value estimate between 0 and 1
        """
        eval_prompt = self._create_evaluation_prompt(state, action)
        
        cache_key = self._create_cache_key(eval_prompt, {})
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
        
        response = self.generate(eval_prompt, temperature=0.3)
        value = self._parse_value_from_response(response)
        
        self.evaluation_cache[cache_key] = value
        
        return value
    
    def get_actions(self, state: Any) -> List[Any]:
        """
        Generate possible actions using LLM.
        
        Args:
            state: Current state
            
        Returns:
            List of possible actions
        """
        action_prompt = self._create_action_prompt(state)
        
        candidates = []
        for i in range(3):
            response = self.generate(
                action_prompt,
                temperature=0.8 + i * 0.1
            )
            actions = self._parse_actions_from_response(response)
            candidates.extend(actions)
        
        seen = set()
        unique_actions = []
        for action in candidates:
            action_str = str(action)
            if action_str not in seen:
                seen.add(action_str)
                unique_actions.append(action)
        
        return unique_actions[:5]
    
    def reflect(self, trajectory: List[Tuple[Any, Any]], outcome: float) -> str:
        """
        Generate reflection on trajectory for improvement.
        
        Args:
            trajectory: List of (state, action) pairs
            outcome: Final outcome value
            
        Returns:
            Reflection text
        """
        reflection_prompt = self._create_reflection_prompt(trajectory, outcome)
        reflection = self.generate(reflection_prompt, temperature=0.5)
        
        return reflection
    
    def _create_evaluation_prompt(self, state: Any, action: Any) -> str:
        """Create prompt for state evaluation."""
        prompt = f"""Evaluate the following state in terms of progress toward the goal.

State: {state}
"""
        if action:
            prompt += f"Action taken: {action}\n"
        
        prompt += """
Provide a detailed analysis considering:
1. How close is this state to achieving the goal?
2. What are the strengths of this position?
3. What challenges remain?

Finally, provide a value score from 0 to 10, where:
- 0 means complete failure or no progress
- 5 means moderate progress
- 10 means goal achieved or excellent progress

Value Score: [YOUR SCORE HERE]
"""
        return prompt
    
    def _create_action_prompt(self, state: Any) -> str:
        """Create prompt for action generation."""
        prompt = f"""Given the current state, generate possible next actions.

Current State: {state}

Generate 3 diverse, high-quality actions that could advance from this state.
Consider different strategies and approaches.

Format your response as a numbered list:
1. [First action]
2. [Second action]
3. [Third action]

Actions:"""
        return prompt
    
    def _create_reflection_prompt(self, trajectory: List[Tuple[Any, Any]], outcome: float) -> str:
        """Create prompt for trajectory reflection."""
        traj_str = "\n".join([f"Step {i+1}: State={s}, Action={a}" 
                             for i, (s, a) in enumerate(trajectory)])
        
        prompt = f"""Analyze the following trajectory and outcome:

Trajectory:
{traj_str}

Final Outcome: {outcome:.2f}

Provide a reflection that:
1. Identifies what worked well
2. Points out mistakes or suboptimal decisions
3. Suggests specific improvements for future attempts

Reflection:"""
        return prompt
    
    def _parse_value_from_response(self, response: str) -> float:
        """Parse value score from LLM response."""
        import re
        
        patterns = [
            r"[Vv]alue [Ss]core:\s*(\d+\.?\d*)",
            r"[Ss]core:\s*(\d+\.?\d*)",
            r"\[(\d+\.?\d*)\]"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                return min(max(score / 10.0, 0.0), 1.0)
        
        self.logger.warning("Could not parse value from response, using default")
        return 0.5
    
    def _parse_actions_from_response(self, response: str) -> List[str]:
        """Parse actions from LLM response."""
        actions = []
        
        import re
        pattern = r"\d+\.\s*(.+)"
        matches = re.findall(pattern, response)
        
        if matches:
            actions = [match.strip() for match in matches]
        else:
            lines = response.strip().split('\n')
            actions = [line.strip() for line in lines if line.strip()]
        
        return actions[:3]
    
    def _mock_llm_generate(self, prompt: str, **kwargs) -> str:
        """
        Mock LLM generation for demonstration.
        Replace with actual API call in production.
        """
        time.sleep(0.01)
        
        if "evaluate" in prompt.lower():
            return """
            This state shows moderate progress toward the goal.
            
            Strengths:
            - Clear structure established
            - Some key elements in place
            
            Challenges:
            - Needs more development
            - Some aspects remain unclear
            
            Value Score: 6.5
            """
        elif "actions" in prompt.lower() or "next" in prompt.lower():
            return """
            1. Expand the current concept with more specific details
            2. Introduce a contrasting element to create tension
            3. Refine and polish the existing structure
            """
        elif "reflection" in prompt.lower():
            return """
            The trajectory shows a reasonable approach but could be optimized.
            Key improvements: More systematic exploration, better evaluation of alternatives.
            """
        else:
            return "Generic response for demonstration purposes."
    
    def _create_cache_key(self, prompt: str, kwargs: Dict) -> str:
        """Create cache key for responses."""
        import hashlib
        key_str = prompt + str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
