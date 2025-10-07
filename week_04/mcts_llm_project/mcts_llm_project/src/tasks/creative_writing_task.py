"""
Creative Writing Task Implementation
Author: Student Name
Course: NEU Self-Improving AI
Date: 2025

Novel application of MCTS+LLM for creative writing optimization.
Different from typical reasoning/math tasks in LATS paper.
"""

from typing import Any, List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import json
import numpy as np
from enum import Enum

from ..mcts.node import MCTSNode
from ..mcts.uct_algorithm import UCTAlgorithm, UCTConfig
from ..llm.model_interface import LanguageModelInterface, LLMConfig


class StoryElement(Enum):
    """Types of story elements."""
    CHARACTER = "character"
    SETTING = "setting"
    CONFLICT = "conflict"
    PLOT_POINT = "plot_point"
    RESOLUTION = "resolution"
    THEME = "theme"


@dataclass
class StoryState:
    """
    Represents the current state of a story being constructed.
    
    Attributes:
        elements: List of story elements added so far
        outline: Current story outline text
        coherence_score: Measure of story coherence (0-1)
        creativity_score: Measure of creativity/originality (0-1)
        depth: Current depth in story construction
        genre: Story genre constraint
        target_length: Target story length
    """
    elements: List[Dict[str, Any]] = field(default_factory=list)
    outline: str = ""
    coherence_score: float = 1.0
    creativity_score: float = 0.5
    depth: int = 0
    genre: str = "fantasy"
    target_length: int = 5  # Number of major story beats
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        """String representation for debugging."""
        return f"StoryState(depth={self.depth}, elements={len(self.elements)}, coherence={self.coherence_score:.2f})"
    
    def to_json(self) -> str:
        """Convert to JSON for serialization."""
        return json.dumps({
            'elements': self.elements,
            'outline': self.outline,
            'coherence_score': self.coherence_score,
            'creativity_score': self.creativity_score,
            'depth': self.depth,
            'genre': self.genre
        })
    
    def is_complete(self) -> bool:
        """Check if story construction is complete."""
        return self.depth >= self.target_length


@dataclass
class StoryAction:
    """
    Represents an action in story construction.
    
    Attributes:
        action_type: Type of story element to add
        content: Specific content of the action
        confidence: Confidence in this action (0-1)
    """
    action_type: StoryElement
    content: str
    confidence: float = 0.5
    
    def __str__(self):
        """String representation."""
        return f"{self.action_type.value}: {self.content[:50]}..."
    
    def __hash__(self):
        """Hash for set operations."""
        return hash((self.action_type, self.content))
    
    def __eq__(self, other):
        """Equality comparison."""
        if not isinstance(other, StoryAction):
            return False
        return self.action_type == other.action_type and self.content == other.content


class CreativeWritingTask:
    """
    Implements creative writing as an MCTS task.
    
    This task uses MCTS with LLM to generate optimal story outlines,
    balancing coherence, creativity, and narrative structure.
    """
    
    def __init__(self, llm_config: LLMConfig, task_config: Optional[Dict] = None):
        """
        Initialize creative writing task.
        
        Args:
            llm_config: Configuration for LLM
            task_config: Optional task-specific configuration
        """
        self.llm = LanguageModelInterface(llm_config)
        self.task_config = task_config or self._default_config()
        
        # Metrics tracking
        self.metrics = {
            'total_stories': 0,
            'average_coherence': 0.0,
            'average_creativity': 0.0,
            'best_story': None,
            'best_score': 0.0
        }
    
    def _default_config(self) -> Dict:
        """Default task configuration."""
        return {
            'genre': 'fantasy',
            'target_length': 5,
            'coherence_weight': 0.6,
            'creativity_weight': 0.4,
            'max_depth': 7,
            'evaluation_criteria': [
                'narrative_coherence',
                'character_development',
                'plot_structure',
                'thematic_depth',
                'originality'
            ]
        }
    
    def get_initial_state(self, prompt: Optional[str] = None) -> StoryState:
        """
        Create initial story state.
        
        Args:
            prompt: Optional prompt to guide story generation
            
        Returns:
            Initial StoryState
        """
        if prompt:
            # Parse prompt for constraints
            genre = self._extract_genre(prompt)
            target_length = self._extract_target_length(prompt)
        else:
            genre = self.task_config['genre']
            target_length = self.task_config['target_length']
        
        return StoryState(
            genre=genre,
            target_length=target_length,
            outline=prompt or f"A {genre} story begins..."
        )
    
    def get_actions(self, state: StoryState) -> List[StoryAction]:
        """
        Generate possible story actions from current state.
        
        Args:
            state: Current story state
            
        Returns:
            List of possible story actions
        """
        if state.is_complete():
            return []
        
        # Determine what type of element to add next
        next_element_type = self._determine_next_element(state)
        
        # Use LLM to generate action candidates
        prompt = self._create_action_generation_prompt(state, next_element_type)
        llm_response = self.llm.generate(prompt, temperature=0.8)
        
        # Parse actions from response
        actions = self._parse_story_actions(llm_response, next_element_type)
        
        return actions
    
    def take_action(self, state: StoryState, action: StoryAction) -> Tuple[StoryState, float, bool]:
        """
        Apply an action to the current state.
        
        Args:
            state: Current story state
            action: Action to apply
            
        Returns:
            Tuple of (new_state, reward, is_terminal)
        """
        # Create new state
        new_state = self._apply_action(state, action)
        
        # Calculate immediate reward
        reward = self._calculate_reward(state, new_state, action)
        
        # Check if terminal
        is_terminal = new_state.is_complete()
        
        return new_state, reward, is_terminal
    
    def evaluate_state(self, state: StoryState) -> float:
        """
        Evaluate the quality of a story state.
        
        Args:
            state: Story state to evaluate
            
        Returns:
            Value estimate (0-1)
        """
        # Use LLM to evaluate story quality
        eval_prompt = self._create_evaluation_prompt(state)
        llm_response = self.llm.generate(eval_prompt, temperature=0.3)
        
        # Parse evaluation scores
        scores = self._parse_evaluation_scores(llm_response)
        
        # Weight scores according to configuration
        weighted_score = (
            self.task_config['coherence_weight'] * scores['coherence'] +
            self.task_config['creativity_weight'] * scores['creativity']
        )
        
        return weighted_score
    
    def run_mcts_search(self, initial_prompt: str, num_simulations: int = 100) -> Dict[str, Any]:
        """
        Run MCTS search to generate optimal story.
        
        Args:
            initial_prompt: Starting prompt for story
            num_simulations: Number of MCTS simulations
            
        Returns:
            Dictionary containing best story and statistics
        """
        # Setup initial state
        initial_state = self.get_initial_state(initial_prompt)
        
        # Configure UCT
        uct_config = UCTConfig(
            num_simulations=num_simulations,
            exploration_constant=1.414,
            simulation_depth_limit=self.task_config.get('max_depth', 7),
            verbose=True
        )
        
        # Create UCT algorithm
        uct = UCTAlgorithm(
            config=uct_config,
            get_actions=self.get_actions,
            take_action=self.take_action,
            evaluate_state=self.evaluate_state
        )
        
        # Run search
        best_action, stats = uct.search(initial_state)
        
        # Build complete story following best path
        story = self._construct_story(initial_state, uct)
        
        # Update metrics
        self._update_metrics(story, stats)
        
        return {
            'story': story,
            'statistics': stats,
            'metrics': self.metrics
        }
    
    def _determine_next_element(self, state: StoryState) -> StoryElement:
        """Determine what type of story element to add next."""
        depth = state.depth
        
        if depth == 0:
            return StoryElement.CHARACTER
        elif depth == 1:
            return StoryElement.SETTING
        elif depth == 2:
            return StoryElement.CONFLICT
        elif depth < state.target_length - 1:
            return StoryElement.PLOT_POINT
        else:
            return StoryElement.RESOLUTION
    
    def _apply_action(self, state: StoryState, action: StoryAction) -> StoryState:
        """Apply action to create new state."""
        new_elements = state.elements.copy()
        new_elements.append({
            'type': action.action_type.value,
            'content': action.content
        })
        
        new_outline = f"{state.outline}\n{action.action_type.value.upper()}: {action.content}"
        
        # Calculate new scores
        coherence = self._calculate_coherence(new_elements)
        creativity = self._calculate_creativity(new_elements)
        
        return StoryState(
            elements=new_elements,
            outline=new_outline,
            coherence_score=coherence,
            creativity_score=creativity,
            depth=state.depth + 1,
            genre=state.genre,
            target_length=state.target_length
        )
    
    def _calculate_reward(self, old_state: StoryState, new_state: StoryState, action: StoryAction) -> float:
        """Calculate immediate reward for taking action."""
        # Reward based on improvement in scores
        coherence_delta = new_state.coherence_score - old_state.coherence_score
        creativity_delta = new_state.creativity_score - old_state.creativity_score
        
        reward = (
            self.task_config['coherence_weight'] * coherence_delta +
            self.task_config['creativity_weight'] * creativity_delta
        )
        
        # Bonus for completing story
        if new_state.is_complete():
            reward += 0.2
        
        return max(0, reward)
    
    def _calculate_coherence(self, elements: List[Dict]) -> float:
        """Calculate story coherence score."""
        if len(elements) <= 1:
            return 1.0
        
        # Simple heuristic: penalize contradictions and reward connections
        # In production, use LLM for sophisticated evaluation
        coherence = 0.9 ** (len(elements) - 1)  # Slight decay with length
        
        # Check for thematic consistency
        themes = [e.get('content', '').lower() for e in elements]
        common_words = set()
        for theme in themes:
            words = set(theme.split())
            if common_words:
                overlap = len(common_words & words) / len(words) if words else 0
                coherence += overlap * 0.1
            common_words.update(words)
        
        return min(1.0, coherence)
    
    def _calculate_creativity(self, elements: List[Dict]) -> float:
        """Calculate creativity/originality score."""
        if not elements:
            return 0.5
        
        # Simple heuristic: reward diversity and unexpected combinations
        # In production, use LLM for sophisticated evaluation
        unique_words = set()
        total_words = 0
        
        for element in elements:
            words = element.get('content', '').split()
            unique_words.update(words)
            total_words += len(words)
        
        diversity = len(unique_words) / total_words if total_words > 0 else 0.5
        
        return min(1.0, diversity)
    
    def _create_action_generation_prompt(self, state: StoryState, element_type: StoryElement) -> str:
        """Create prompt for generating story actions."""
        return f"""
        Current Story Outline:
        {state.outline}
        
        Genre: {state.genre}
        Current Elements: {len(state.elements)}
        
        Generate 3 different options for the next {element_type.value} to add to this story.
        Each option should be creative, coherent with the existing story, and advance the narrative.
        
        Format:
        1. [First option]
        2. [Second option]
        3. [Third option]
        """
    
    def _create_evaluation_prompt(self, state: StoryState) -> str:
        """Create prompt for evaluating story state."""
        return f"""
        Evaluate the following story outline:
        
        {state.outline}
        
        Genre: {state.genre}
        
        Rate the story on:
        1. Narrative Coherence (0-10): How well do the elements connect?
        2. Creativity/Originality (0-10): How unique and engaging is the story?
        3. Structure (0-10): How well-structured is the narrative?
        
        Provide scores in the format:
        Coherence: [score]
        Creativity: [score]
        Structure: [score]
        """
    
    def _parse_story_actions(self, response: str, element_type: StoryElement) -> List[StoryAction]:
        """Parse story actions from LLM response."""
        actions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 4)):
                # Remove numbering
                content = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                action = StoryAction(
                    action_type=element_type,
                    content=content,
                    confidence=0.7
                )
                actions.append(action)
        
        return actions[:3]
    
    def _parse_evaluation_scores(self, response: str) -> Dict[str, float]:
        """Parse evaluation scores from LLM response."""
        scores = {
            'coherence': 0.5,
            'creativity': 0.5,
            'structure': 0.5
        }
        
        import re
        
        patterns = {
            'coherence': r'[Cc]oherence:\s*(\d+(?:\.\d+)?)',
            'creativity': r'[Cc]reativity:\s*(\d+(?:\.\d+)?)',
            'structure': r'[Ss]tructure:\s*(\d+(?:\.\d+)?)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                scores[key] = min(1.0, score / 10.0)  # Normalize to 0-1
        
        return scores
    
    def _construct_story(self, initial_state: StoryState, uct: UCTAlgorithm) -> Dict[str, Any]:
        """Construct complete story following best path."""
        # Simplified: return initial state with metadata
        # In full implementation, follow best path through tree
        return {
            'outline': initial_state.outline,
            'elements': initial_state.elements,
            'genre': initial_state.genre,
            'final_scores': {
                'coherence': initial_state.coherence_score,
                'creativity': initial_state.creativity_score
            }
        }
    
    def _update_metrics(self, story: Dict, stats: Dict) -> None:
        """Update task metrics."""
        self.metrics['total_stories'] += 1
        
        # Update averages
        if 'final_scores' in story:
            self.metrics['average_coherence'] = (
                (self.metrics['average_coherence'] * (self.metrics['total_stories'] - 1) +
                 story['final_scores']['coherence']) / self.metrics['total_stories']
            )
            self.metrics['average_creativity'] = (
                (self.metrics['average_creativity'] * (self.metrics['total_stories'] - 1) +
                 story['final_scores']['creativity']) / self.metrics['total_stories']
            )
            
            # Track best story
            total_score = (story['final_scores']['coherence'] + 
                          story['final_scores']['creativity']) / 2
            if total_score > self.metrics['best_score']:
                self.metrics['best_score'] = total_score
                self.metrics['best_story'] = story
    
    def _extract_genre(self, prompt: str) -> str:
        """Extract genre from prompt."""
        genres = ['fantasy', 'sci-fi', 'mystery', 'romance', 'thriller']
        prompt_lower = prompt.lower()
        
        for genre in genres:
            if genre in prompt_lower:
                return genre
        
        return 'fantasy'  # Default
    
    def _extract_target_length(self, prompt: str) -> int:
        """Extract target length from prompt."""
        # Simple extraction, could be improved
        if 'short' in prompt.lower():
            return 3
        elif 'long' in prompt.lower():
            return 7
        else:
            return 5
    
    
    