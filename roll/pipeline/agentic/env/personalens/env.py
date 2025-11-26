import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gem import Env

from roll.pipeline.agentic.utils import all_seed
from roll.utils.logging import get_logger

logger = get_logger()

# Personalization evaluation prompt template (from PersonaLens original project)
EVAL_DIALOGUE_PERSONALIZATION_PROMPT = """<task_description>
Evaluate the degree to which a conversation between a USER and an ASSISTANT aligns with personalization by assessing how well the assistant learns from, remembers, and proactively applies user preferences and patterns.
</task_description>

<definitions>
- Score: A rating from 1-4 (1=Poor, 4=Excellent).
- User Demographic Profile: The user's demographic information.
- User Preferences: The user's relevant preferences.
- Explicit Preferences: Preferences clearly stated by the user
- Implicit Preferences: Preferences inferred from patterns, habits, contextual clues, past interactions or user behavior.
- User Control: The level of influence the user has in making decisions or directing the course of an interaction.
- Past Interaction Summary: A summary of relevant past user interactions.
- Task Description: The description of the task the user needs help with.
- Current Situation Context: The user's current situation.
- Conversation: A sequence of USER inputs and ASSISTANT responses.
</definitions>

<instructions>
1. Evaluate the conversation against these key criteria:
- Proactive Learning: Does the assistant demonstrate learning from past interactions?
- Preference Application: Does the assistant proactively apply user preferences?
- Contextual Awareness: Does the assistant adapt to user's current situation?
- User Agency: Does the assistant maintain user control while showing personalization?

2. Score using the following guidelines:

Score of 1: POOR (Complete Failure to Personalize)
- The assistant fails to apply known preferences that should be automatically recalled from past interactions.
- The assistant asks for basic information that should already be known, such as the time of the alarm or sound preference, when those preferences have already been established.
- The assistant contradicts previously established preferences or gives responses that are inconsistent with the user's history.
- There is no learning from past interactions, and the assistant does not personalize the experience in any meaningful way.

<example>
Past Interaction Summary: The user sets a 7am weekday alarm with upbeat music.
USER: "Hey there, I need to set my weekday alarm for tomorrow."
ASSISTANT: "I can help you set an alarm. What time would you like it set for?"
USER: "7am like usual please."
ASSISTANT: "Would you like this to be a one-time alarm or recurring?"
USER: "Recurring for weekdays, just like I always do."
ASSISTANT: "What sound would you like for the alarm?"
USER: "The same upbeat music I always use."
ASSISTANT: "I'll need you to specify which sound you'd like from our available options."

Personalization Score: 1
Justification: Despite multiple interactions where the user references established patterns ("like usual", "always do", "always use"), the assistant fails to recognize or apply any previous preferences. Each aspect of the alarm setting requires explicit input, creating a frustrating experience that shows no learning or personalization.
</example>

Score of 2: BASIC (Minimal Personalization)
- The assistant acknowledges user preferences only when explicitly stated in the current conversation.
- The assistant requires explicit restatement of preferences that have already been established in past interactions.
- Implicit preferences are missed or not applied unless explicitly mentioned by the user.
- The assistant may suggest minimal changes or adjustments based on the current conversation, but it does not proactively personalize the experience.

<example>
Past Interaction Summary: Uses gentle nature sounds, hates loud beeping
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "What time would you like the alarm set for?"
USER: "7am like always."
ASSISTANT: "Ah yes, I see you usually set it for 7am. Would you like the same upbeat music as your alarm sound?"
USER: "Yes, please."
ASSISTANT: "I'll set that up. Should I make it recurring for weekdays as usual?"
USER: "Yes, that would be great."
ASSISTANT: "Okay, I've set your 7am weekday alarm with the upbeat music."

Personalization Score: 2
Justification: The assistant shows minimal recognition of preferences only after they're mentioned by the user. While it acknowledges patterns once prompted, it still requires initial user input rather than proactively applying known preferences from past interaction summary.
</example>


Score of 3: STRONG (Proactive Personalization)
- The assistant proactively applies known preferences from past interactions without needing explicit user input.
- It applies learned preferences from previous interactions but might still ask for minor adjustments (e.g., if the user wants to change something).
- Successfully identifies implicit preferences
- Maintains user agency while showing knowledge
- Makes intelligent suggestions based on context

<example>
Past Interaction Summary: 7am weekday alarm with gradual volume increase
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "I've set your usual 7am weekday alarm with your preferred upbeat music. Since tomorrow's Monday, would you like me to add your occasional backup alarm?"
USER: "Actually, yes please. That would be helpful."
ASSISTANT: "Done! I've added a backup alarm for 7:15am. I've also noticed you've been enjoying energetic pop songs lately - would you like to try a new wake-up song I can recommend?"
USER: "Sure, what do you have in mind?"
ASSISTANT: "Based on your positive feedback about previous morning songs, I think you'd enjoy this new upbeat track. Should I set it as your alarm sound for tomorrow?"

Personalization Score: 3
Justification: The assistant proactively applies known preferences while offering relevant customizations based on patterns and context. It demonstrates learning from past interactions and makes intelligent suggestions while maintaining user control.
</example>

Score of 4: EXCEPTIONAL (Perfect Personalization)
- The assistant anticipates user needs based on both explicit and implicit preferences.
- It applies sophisticated understanding of the user's habits, identifying patterns, and proactively adjusting for future needs.
- The assistant doesn't simply rely on explicit preferences, it recognizes context and makes intelligent suggestions based on its deep knowledge of the user's habits.

<example>
Past Interaction Summary: 7am weekdays, 8am weekends, takes longer to wake up on Mondays
USER: "I need to set my weekday alarm for tomorrow."
ASSISTANT: "I've already set your standard 7am alarm with the upbeat music for tomorrow. Since it's Sunday evening in Toronto and tomorrow's forecast shows rain, which typically affects your commute time, I've also added a 6:45am backup alarm. I noticed this follows your pattern of preferring earlier starts on rainy Mondays."
USER: "That's really thoughtful, thanks!"
ASSISTANT: "You're welcome! I also see you have a 9am presentation tomorrow from your calendar. On previous presentation days, you've mentioned preferring extra time to prepare, so I've included a gentle reminder alarm at 6:30am. Would you like to keep these adjustments?"
USER: "Yes, that's perfect. Thanks for thinking ahead!"
ASSISTANT: "Great! And since you've shown interest in varying your wake-up music, I've queued three new upbeat songs similar to your previous favorites. They'll play in sequence if you hit snooze, which your pattern shows happens more often on presentation days."

Personalization Score: 4
Justification: The assistant demonstrates exceptional personalization by combining multiple layers of context (weather, calendar events, past behaviors) with learned preferences. It anticipates needs based on patterns, considers environmental factors, and provides sophisticated customization while maintaining user agency. The assistant shows deep understanding of the user's habits and proactively offers relevant, contextual adjustments while explaining its reasoning.
</example>

3. Review provided context information:
<user_demographic_profile>
{demographic_profile}
</user_demographic_profile>

<user_preferences>
{user_affinity}
</user_preferences>

<task_description>
{task_description}
</task_description>

<past_interaction_summary>
{interaction_summary}
</past_interaction_summary>

<current_situation_context>
{situation_context}
</current_situation_context>

<conversation>
{conversation}
</conversation>

4. Provide your evaluation score and justification in the following format:

<response_format>
Personalization Score: [1-4]
Key Observations:
- [List key aspects of personalization or lack thereof]
- [Note specific examples of proactive or reactive behavior]
- [Identify missed opportunities]

Justification: [Detailed explanation of score based on criteria]

Improvement Suggestions:
- [Specific ways the response could be more personalized]
</response_format>

5. Additional Guidelines
- Evaluate based on all available context information
- Consider both explicit and implicit preferences
- Assess balance between personalization and user control
- Look for evidence of learning and pattern recognition
- Consider appropriateness of personalization level for context
</instructions>

<response>
Provide your response immediately without any preamble, enclosed in <response></response> tags.
</response>
"""


class PersonaLensEnv(Env):
    """
    PersonaLens environment for personalized conversational AI assistant evaluation.

    This environment loads user profiles and tasks from PersonaLens dataset and
    simulates task-oriented dialogues between a user and an AI assistant.
    """
    def __init__(self,
                 data_dir: str,
                 max_steps: int = 20,
                 format_penalty: float = 0.0,
                 action_pattern: str = "<answer>(.*?)</answer>",
                 special_token_list: Tuple[str, ...] = ("<|im_start|>", "<|im_end|>"),
                 user_id: Optional[int] = None,
                 task_id: Optional[str] = None,
                 seed: Optional[int] = None,
                 include_demographic: bool = True,
                 include_past_interaction: bool = True,
                 include_situation: bool = True,
                 use_llm_judge: bool = False,
                 judge_llm_config: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize PersonaLens environment.

        Args:
            data_dir: Path to PersonaLens data directory (should contain profile/ and task/ subdirectories)
            max_steps: Maximum number of dialogue turns
            format_penalty: Penalty for invalid action format
            action_pattern: Regex pattern to extract action from model response
            special_token_list: List of special tokens to remove from responses
            user_id: Specific user ID to use (if None, randomly selects)
            task_id: Specific task ID to use (if None, randomly selects from user's tasks)
            seed: Random seed
            include_demographic: Whether to include demographic information in context
            include_past_interaction: Whether to include past interaction summary
            include_situation: Whether to include situational context
            use_llm_judge: Whether to use LLM judge for personalization evaluation (default: False, uses keyword matching)
            judge_llm_config: Dict with LLM judge configuration. Required keys:
                - api_key: API key for LLM service
                - base_url: Base URL for LLM API (default: "https://api.openai.com/v1")
                - model_name: Model name to use (default: "gpt-4")
                - timeout: Request timeout in seconds (default: 60)
        """
        Env.__init__(self)

        self.data_dir = Path(data_dir)
        self.max_steps = max_steps
        self.format_penalty = format_penalty
        self.action_pattern = action_pattern
        self.special_token_list = special_token_list

        self.include_demographic = include_demographic
        self.include_past_interaction = include_past_interaction
        self.include_situation = include_situation
        
        # LLM Judge configuration
        self.use_llm_judge = use_llm_judge
        self.judge_llm_config = judge_llm_config or {}
        self.judge_client = None
        
        if self.use_llm_judge:
            self._init_llm_judge()
        
        # Load available users and tasks
        self._load_user_list()

        # State variables
        self.current_user_id = user_id
        self.current_task_id = task_id
        self.user_profile = None
        self.task_info = None
        self.conversation_history: List[Dict[str, str]] = []
        self.num_steps = 0
        self.task_completed = False

        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _init_llm_judge(self):
        """Initialize LLM judge client for personalization evaluation."""
        try:
            from openai import OpenAI
            
            api_key = self.judge_llm_config.get("api_key", None)
            base_url = self.judge_llm_config.get("base_url", "https://api.openai.com/v1")
            model_name = self.judge_llm_config.get("model_name", "gpt-4")
            timeout = self.judge_llm_config.get("timeout", 60)
            
            if not api_key:
                logger.warning("LLM judge api_key not provided, falling back to keyword matching")
                self.use_llm_judge = False
                return
            
            self.judge_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
            self.judge_model_name = model_name
            logger.info(f"LLM judge initialized: base_url={base_url}, model={model_name}")
        except ImportError:
            logger.warning("OpenAI package not available, falling back to keyword matching")
            self.use_llm_judge = False
        except Exception as e:
            logger.warning(f"Failed to initialize LLM judge: {e}, falling back to keyword matching")
            self.use_llm_judge = False
    
    def _load_user_list(self):
        """Load list of available user IDs from data directory."""
        profile_dir = self.data_dir / "profile"
        if not profile_dir.exists():
            raise ValueError(f"Profile directory not found: {profile_dir}")

        user_dirs = [d for d in profile_dir.iterdir() if d.is_dir() and d.name.startswith("user")]
        self.available_users = sorted([int(d.name.replace("user", "")) for d in user_dirs])

        if len(self.available_users) == 0:
            raise ValueError(f"No user profiles found in {profile_dir}")

    def _load_user_profile(self, user_id: int) -> Dict:
        """Load user profile from JSON file."""
        profile_file = self.data_dir / "profile" / f"user{user_id}" / "profile.json"
        if not profile_file.exists():
            raise ValueError(f"Profile file not found: {profile_file}")

        with open(profile_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_user_tasks(self, user_id: int, multi_domain: bool = False) -> Dict:
        """Load user tasks from JSON file."""
        task_file = self.data_dir / "profile" / f"user{user_id}" / ("tasks_md.json" if multi_domain else "tasks.json")
        if not task_file.exists():
            raise ValueError(f"Task file not found: {task_file}")

        with open(task_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _format_user_context(self) -> str:
        """Format user context information for the assistant."""
        context_parts = []

        # Use 'demographics' (plural) as in original PersonaLens
        if self.include_demographic and "demographics" in self.user_profile:
            demo = self.user_profile["demographics"]
            demo_str = "\n".join([f"- {k}: {v}" for k, v in demo.items()])
            context_parts.append(f"User Demographic Information:\n{demo_str}")

        # Use 'interactions[domain]' as in original PersonaLens
        if self.include_past_interaction:
            relevant_domain = self.task_info.get("Relevant Domains", [""])[0]
            if relevant_domain and "interactions" in self.user_profile:
                if relevant_domain in self.user_profile["interactions"]:
                    summary = self.user_profile["interactions"][relevant_domain]
                    context_parts.append(f"User Past Interaction Summary:\n{summary}")

        if self.include_situation and "situations" in self.task_info:
            situation = self.task_info["situations"]
            situation_str = "\n".join([f"- {k}: {v}" for k, v in situation.items()])
            context_parts.append(f"Current Situation Context:\n{situation_str}")

        return "\n\n".join(context_parts)

    def _format_task_description(self) -> str:
        """Format task description and instructions."""
        task_desc = self.task_info.get("Task Description", "")
        user_intent = self.task_info.get("User Intent", "")
        task_goal = self.task_info.get("Task Goal", "")

        # Format relevant affinities
        relevant_domain = self.task_info.get("Relevant Domains", [""])[0]
        relevant_affinities = self.task_info.get("Relevant Affinity Types", [])

        affinity_info = ""
        if relevant_domain in self.user_profile.get("affinities", {}):
            domain_affinities = self.user_profile["affinities"][relevant_domain]
            relevant_info = {}
            for aff_type in relevant_affinities:
                if aff_type in domain_affinities:
                    relevant_info[aff_type] = domain_affinities[aff_type]

            if relevant_info:
                affinity_str = "\n".join([
                    f"- {k}: {', '.join(map(str, v))}" if isinstance(v, list) else f"- {k}: {v}"
                    for k, v in relevant_info.items()
                ])
                affinity_info = f"\n\nUser Preferences (Relevant to this task):\n{affinity_str}"

        task_str = f"""Task Description: {task_desc}

User Intent: {user_intent}

Task Goal: {task_goal}{affinity_info}

You are a conversational AI assistant. Your goal is to help the user complete this task in a personalized and natural way. Consider the user's preferences and context when providing assistance."""

        return task_str

    def get_instructions(self) -> str:
        """Get environment instructions for the agent."""
        user_context = self._format_user_context()
        task_description = self._format_task_description()

        instruction = f"""{task_description}

{user_context}

Conversation History:
{self._format_conversation_history()}

You should provide helpful, personalized responses to help the user complete the task. 
Respond naturally and conversationally. The user may say "TERMINATE" when they are satisfied with the task completion.
Your response should be conversational text, not action commands."""

        return instruction

    def _format_conversation_history(self) -> str:
        """Format conversation history as text."""
        if not self.conversation_history:
            return "No conversation history yet."

        history_lines = []
        for msg in self.conversation_history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines)

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environment to start a new task.

        Args:
            seed: Random seed for selecting user/task

        Returns:
            observation: Initial observation (task description and context)
            info: Additional information dictionary
        """
        Env.reset(self, seed)

        try:
            with all_seed(seed):
                # Select user if not specified
                if self.current_user_id is None:
                    self.current_user_id = random.choice(self.available_users)

                # Load user profile
                self.user_profile = self._load_user_profile(self.current_user_id)

                # Load tasks
                tasks = self._load_user_tasks(self.current_user_id)

                # Select task if not specified
                if self.current_task_id is None:
                    task_keys = list(tasks.keys())
                    task_key = random.choice(task_keys)
                else:
                    # Find task by task_id
                    task_key = None
                    for k, v in tasks.items():
                        if v.get("task_id") == self.current_task_id:
                            task_key = k
                            break
                    if task_key is None:
                        raise ValueError(f"Task ID {self.current_task_id} not found for user {self.current_user_id}")

                self.task_info = tasks[task_key]
                self.current_task_id = self.task_info.get("task_id", task_key)

                # Reset conversation state
                self.conversation_history = []
                self.num_steps = 0
                self.task_completed = False

                # Generate initial user query (simplified - in real PersonaLens this would use LLM)
                initial_query = self._generate_initial_user_query()
                self.conversation_history.append({"role": "user", "content": initial_query})

                observation = self.get_instructions()
                info = {
                    "env_instruction": observation,
                    "user_id": self.current_user_id,
                    "task_id": self.current_task_id,
                    "conversation_history": self.conversation_history.copy()
                }

                return observation, info

        except (RuntimeError, RuntimeWarning, ValueError):
            # Retry with different seed
            next_seed = abs(hash(str(seed))) % (2**32) if seed is not None else None
            return self.reset(next_seed)

    def _generate_initial_user_query(self) -> str:
        """Generate initial user query based on task (simplified version)."""
        user_intent = self.task_info.get("User Intent", "")
        # In real PersonaLens, this would use an LLM to generate a natural query
        # For now, we use a simplified version based on User Intent
        if user_intent:
            # Extract the core intent and format as a question/request
            intent_lower = user_intent.lower()
            if "find" in intent_lower or "recommend" in intent_lower:
                return f"Can you help me {user_intent.lower()}?"
            elif "set" in intent_lower or "create" in intent_lower:
                return f"I need to {user_intent.lower()}. Can you assist?"
            else:
                return f"{user_intent}. Can you help me with this?"
        return "Hello, I need some assistance."
    
    def _evaluate_personalization_with_keywords(self, assistant_response: str) -> float:
        """Evaluate personalization score using keyword matching (fallback method)."""
        personalization_score = 0.0
        if relevant_domain := self.task_info.get("Relevant Domains", [""])[0]:
            if relevant_domain in self.user_profile.get("affinities", {}):
                domain_affinities = self.user_profile["affinities"][relevant_domain]
                affinity_keywords = []
                for aff_type, aff_value in domain_affinities.items():
                    if isinstance(aff_value, list):
                        affinity_keywords.extend([str(v).lower() for v in aff_value])
                    else:
                        affinity_keywords.append(str(aff_value).lower())
                
                response_lower = assistant_response.lower()
                matched_keywords = sum(1 for kw in affinity_keywords if kw in response_lower)
                if matched_keywords > 0:
                    personalization_score = min(1.0, matched_keywords / max(1, len(affinity_keywords)))
        return personalization_score
    
    def _evaluate_personalization_with_llm(self, assistant_response: str) -> float:
        """Evaluate personalization score using LLM judge (following original PersonaLens format exactly)."""
        try:
            # Format user context (following original PersonaLens evaluate_dialogue.py format)
            # Use 'demographics' (plural) as in original - line 274
            demo_str = ""
            if self.include_demographic and "demographics" in self.user_profile:
                demo = self.user_profile["demographics"]
                demo_str = "\n".join([f"- {k}: {v}" for k, v in demo.items()])
            
            # Format user preferences (following original format - line 308)
            pref_str = ""
            relevant_domain = self.task_info.get("Relevant Domains", [""])[0]
            if relevant_domain and relevant_domain in self.user_profile.get("affinities", {}):
                domain_affinities = self.user_profile["affinities"][relevant_domain]
                pref_str = "\n".join([
                    f"- {k}: {', '.join(map(str, v))}" if isinstance(v, list) else f"- {k}: {v}"
                    for k, v in domain_affinities.items()
                ])
            
            # Format situation context (following original format - line 282)
            situation_str = ""
            if self.include_situation and "situations" in self.task_info:
                situation = self.task_info["situations"]
                situation_str = "\n".join([f"- {k}: {v}" for k, v in situation.items()])
            
            # Format conversation (following original format - line 285-286)
            conversation_str = "\n".join([
                f"[{msg.get('role', 'unknown').upper()}]: {msg.get('content', '')}"
                for msg in self.conversation_history
            ])
            
            # Format interaction summary (following original format - line 305)
            interaction_summary = ""
            if relevant_domain and "interactions" in self.user_profile:
                if relevant_domain in self.user_profile["interactions"]:
                    interaction_summary = self.user_profile["interactions"][relevant_domain]
            if not interaction_summary:
                interaction_summary = "No past interactions available."
            
            # Format task description (following original format - line 276, uses User Intent)
            task_description = self.task_info.get("User Intent", self.task_info.get("Task Description", ""))
            
            # Format the prompt
            prompt = EVAL_DIALOGUE_PERSONALIZATION_PROMPT.format(
                demographic_profile=demo_str or "Not available",
                user_affinity=pref_str or "Not available",
                task_description=task_description,
                interaction_summary=interaction_summary,
                situation_context=situation_str or "Not available",
                conversation=conversation_str
            )
            
            # Call LLM judge
            response = self.judge_client.chat.completions.create(
                model=self.judge_model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for conversational AI personalization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            judge_response = response.choices[0].message.content
            
            # Extract score from response
            score_match = re.search(r"Personalization Score:\s*(\d+)", judge_response, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
                # Normalize to 0-1 range (score is 1-4)
                return (score - 1) / 3.0
            else:
                logger.warning(f"Could not extract score from judge response: {judge_response[:200]}")
                # Fallback to keyword matching
                return self._evaluate_personalization_with_keywords(assistant_response)
                
        except Exception as e:
            logger.warning(f"LLM judge evaluation failed: {e}, falling back to keyword matching")
            return self._evaluate_personalization_with_keywords(assistant_response)

    def step(self, action: str):
        """
        Execute one step in the environment.

        Args:
            action: The assistant's response text

        Returns:
            observation: Updated observation
            reward: Reward for this step
            terminated: Whether the episode is terminated
            truncated: Whether the episode was truncated
            info: Additional information dictionary
        """
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "task_completed": "last",
            "format_penalty": "mean",
            "personalization_score": "mean",
        }

        # Parse action (though for PersonaLens, action is just text response)
        action_info = self.parse_action(action)

        # Check if action is empty (invalid)
        if not action_info["action"] or not action_info["action"].strip():
            # Invalid format - empty response
            reward = self.format_penalty
            self.num_steps += 1

            action_desc = f"At turn {self.num_steps}, you did not provide a valid response."
            metrics = {
                "action_is_valid": False,
                "task_completed": False,
                "format_penalty": self.format_penalty,
                "personalization_score": 0.0
            }
            info = {
                "metrics": metrics,
                "metrics_agg_mode": metrics_agg_mode,
                "action_desc": action_desc
            }
            info.update(action_info)

            observation = self.get_instructions()
            terminated = self.num_steps >= self.max_steps
            truncated = False

            return observation, reward, terminated, truncated, info

        # Add assistant response to conversation history
        assistant_response = action_info["action_content"] or action_info["action"]
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        self.num_steps += 1

        # Check if user wants to terminate
        # In real PersonaLens, this would be determined by a user LLM agent
        # For now, we use heuristics based on conversation length and content
        terminated = False
        task_completed = False

        # Simple heuristic: if conversation is long enough and contains completion signals
        if self.num_steps >= 3:  # Minimum conversation length
            # Check last few messages for completion signals
            recent_messages = " ".join([msg.get("content", "") for msg in self.conversation_history[-4:]])
            completion_signals = ["thank", "perfect", "great", "done", "complete", "satisfied", "terminate"]
            if any(signal in recent_messages.lower() for signal in completion_signals):
                task_completed = True
                terminated = True

        if self.num_steps >= self.max_steps:
            terminated = True

        # Calculate reward (simplified - in real PersonaLens this would use a judge agent)
        reward = 0.0
        if task_completed:
            reward = 1.0
        elif self.num_steps < self.max_steps:
            reward = 0.1  # Small reward for continuing the conversation

        # Calculate personalization score using LLM judge or keyword matching
        if self.use_llm_judge and self.judge_client:
            personalization_score = self._evaluate_personalization_with_llm(assistant_response)
        else:
            personalization_score = self._evaluate_personalization_with_keywords(assistant_response)

        action_desc = f"At turn {self.num_steps}, you responded: {assistant_response[:100]}..."
        if task_completed:
            action_desc += " Task appears to be completed."

        metrics = {
            "action_is_valid": True,
            "task_completed": task_completed,
            "format_penalty": 0,
            "personalization_score": personalization_score
        }

        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode,
            "action_desc": action_desc,
            "conversation_history": self.conversation_history.copy()
        }
        info.update(action_info)

        observation = self.get_instructions()
        truncated = False

        return observation, reward, terminated, truncated, info

    def parse_action(self, text: str):
        """Parse action from text response."""
        # For PersonaLens, we accept any text as valid action
        # But we still use the parser to extract content from formatted responses
        if self.special_token_list is not None:
            for special_token in self.special_token_list:
                text = text.replace(special_token, "").strip()

        # Try to extract content from formatted response
        match = re.search(self.action_pattern, text, re.DOTALL)
        if match:
            action_content = match.group(1).strip() if len(match.groups()) >= 1 else text
        else:
            # If no match, use the entire text (cleaned)
            action_content = text.strip()

        return {
            "action": action_content,  # For PersonaLens, action is the text response
            "action_content": action_content,
            "think_content": ""
        }

    def render(self, mode: str = "text"):
        """Render the current state of the environment."""
        if mode == "text":
            return self._format_conversation_history()
        else:
            raise ValueError(f"Invalid render mode: {mode}")

    def sample_random_action(self):
        """Sample a random action (for testing purposes)."""
        responses = [
            "I can help you with that.",
            "Let me assist you with this task.",
            "Sure, I'd be happy to help.",
            "I understand what you need."
        ]
        return random.choice(responses)

    def close(self):
        """Clean up environment resources."""
        self.conversation_history = []
        self.user_profile = None
        self.task_info = None
