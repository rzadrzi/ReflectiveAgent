"""Prompt builder for constructing LLM prompts."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from reflective_agent.utils.helpers import load_yaml
from reflective_agent.utils.logging import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Builder for constructing prompts from templates."""

    def __init__(self, prompts_config_path: Optional[str] = None):
        """
        Initialize prompt builder.

        Args:
            prompts_config_path: Path to prompts YAML file
        """
        if prompts_config_path is None:
            prompts_config_path = "./configs/prompts.yaml"

        self.prompts_config = load_yaml(prompts_config_path)
        self.prompts = self.prompts_config.get("prompts", {})
        self.strategies = self.prompts_config.get("strategies", {})

        logger.info(f"Loaded {len(self.prompts)} prompt templates")

    def build_system_prompt(
        self,
        base_key: str = "system.base",
        adaptive_additions: Optional[List[str]] = None,
    ) -> str:
        """
        Build system prompt with optional adaptive additions.

        Args:
            base_key: Key for base system prompt in config
            adaptive_additions: List of additional instructions to append

        Returns:
            Complete system prompt
        """
        # Get base prompt
        keys = base_key.split(".")
        prompt_data = self.prompts
        for key in keys:
            prompt_data = prompt_data.get(key, {})

        base_prompt = prompt_data if isinstance(prompt_data, str) else ""

        # Add adaptive additions
        if adaptive_additions:
            additions_text = "\n\n".join(adaptive_additions)
            base_prompt = f"{base_prompt}\n\n## Additional Guidelines\n{additions_text}"

        return base_prompt

    def build_puzzle_prompt(
        self,
        puzzle_text: str,
        puzzle_type: str,
        difficulty: str = "medium",
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict[str, str]]:
        """
        Build messages for puzzle solving.

        Args:
            puzzle_text: The puzzle to solve
            puzzle_type: Type of puzzle (e.g., "sudoku", "logic_riddle")
            difficulty: Difficulty level
            system_prompt: System prompt (optional, uses default if None)
            **kwargs: Additional variables for template

        Returns:
            List of message dictionaries
        """
        # Build system prompt if not provided
        if system_prompt is None:
            system_prompt = self.build_system_prompt()

        # Get puzzle prompt template
        puzzle_template = self.prompts.get("puzzle", {}).get("template", "")

        # Format template
        variables = {
            "system_prompt": system_prompt,
            "puzzle_type": puzzle_type,
            "difficulty": difficulty,
            "puzzle_text": puzzle_text,
            **kwargs,
        }

        try:
            formatted_prompt = puzzle_template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in puzzle template: {e}")
            raise

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt},
        ]

        return messages

    def build_reflection_prompt(
        self,
        puzzle_text: str,
        reasoning_steps: List[str],
        your_answer: str,
        correct_answer: str,
    ) -> List[Dict[str, str]]:
        """
        Build messages for self-reflection.

        Args:
            puzzle_text: The original puzzle
            reasoning_steps: Agent's reasoning steps
            your_answer: Agent's answer
            correct_answer: Correct answer

        Returns:
            List of message dictionaries
        """
        # Get reflection prompt template
        reflection_template = self.prompts.get("reflection", {}).get("template", "")

        # Format variables
        variables = {
            "puzzle_text": puzzle_text,
            "reasoning_steps": "\n".join(f"- {step}" for step in reasoning_steps),
            "your_answer": your_answer,
            "correct_answer": correct_answer,
        }

        try:
            formatted_prompt = reflection_template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in reflection template: {e}")
            raise

        messages = [
            {"role": "user", "content": formatted_prompt},
        ]

        return messages

    def build_debate_prompt(
        self,
        puzzle_text: str,
        strategy_name: str,
        system_prompt: Optional[str] = None,
        is_initial: bool = True,
        your_reasoning: Optional[List[str]] = None,
        other_arguments: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build messages for multi-agent debate.

        Args:
            puzzle_text: The puzzle to solve
            strategy_name: Name of reasoning strategy
            system_prompt: System prompt (optional)
            is_initial: Whether this is the initial debate round
            your_reasoning: Your previous reasoning (for subsequent rounds)
            other_arguments: Other agents' arguments (for subsequent rounds)

        Returns:
            List of message dictionaries
        """
        # Build system prompt
        if system_prompt is None:
            system_prompt = self.build_system_prompt()

        # Get strategy description
        strategy = self.strategies.get(strategy_name, {})
        strategy_description = strategy.get("description", "")

        if is_initial:
            # Initial debate prompt
            template = self.prompts.get("debate", {}).get("initial", "")

            variables = {
                "system_prompt": system_prompt,
                "strategy_name": strategy.get("name", strategy_name),
                "strategy_description": strategy_description,
                "puzzle_text": puzzle_text,
            }
        else:
            # Response to other agents
            template = self.prompts.get("debate", {}).get("response_to_others", "")

            # Format other arguments
            other_args_text = ""
            if other_arguments:
                for i, arg in enumerate(other_arguments, 1):
                    agent_name = arg.get("agent_name", f"Agent {i}")
                    reasoning = "\n".join(f"  - {step}" for step in arg.get("reasoning", []))
                    answer = arg.get("answer", "N/A")
                    other_args_text += (
                        f"\n### {agent_name}\n**Reasoning:**\n{reasoning}\n**Answer:** {answer}\n"
                    )

            variables = {
                "puzzle_text": puzzle_text,
                "your_reasoning": "\n".join(f"- {step}" for step in (your_reasoning or [])),
                "other_arguments": other_args_text,
            }

        try:
            formatted_prompt = template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in debate template: {e}")
            raise

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt},
        ]

        return messages

    def build_arbitration_prompt(
        self,
        puzzle_text: str,
        correct_answer: str,
        agent_solutions: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Build messages for arbitration.

        Args:
            puzzle_text: The original puzzle
            correct_answer: Correct answer
            agent_solutions: List of agent solutions

        Returns:
            List of message dictionaries
        """
        # Get arbitration prompt template
        template = self.prompts.get("arbitration", {}).get("template", "")

        # Format agent solutions
        solutions_text = ""
        for i, solution in enumerate(agent_solutions, 1):
            agent_name = solution.get("agent_name", f"Agent {i}")
            reasoning = "\n".join(f"  - {step}" for step in solution.get("reasoning", []))
            answer = solution.get("answer", "N/A")
            confidence = solution.get("confidence", 0.0)

            solutions_text += f"\n### {agent_name}\n**Reasoning:**\n{reasoning}\n**Answer:** {answer}\n**Confidence:** {confidence}\n"

        variables = {
            "puzzle_text": puzzle_text,
            "correct_answer": correct_answer,
            "agent_solutions": solutions_text,
        }

        try:
            formatted_prompt = template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing variable in arbitration template: {e}")
            raise

        messages = [
            {"role": "user", "content": formatted_prompt},
        ]

        return messages

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available reasoning strategies.

        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())

    def get_strategy_description(self, strategy_name: str) -> str:
        """
        Get description for a reasoning strategy.

        Args:
            strategy_name: Name of strategy

        Returns:
            Strategy description
        """
        strategy = self.strategies.get(strategy_name, {})
        return strategy.get("description", "")


# from typing import Any, Dict, Optional

# class PromptBuilder:
#     def __init__(self, prompts: Dict[str, str]) :
#         self.prompts = prompts

#     def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
#         if context is None:
#             return "No context provided."
#         lines = []

#         for key, value in context.items():
#             lines.append(f"-{key}: {value}")

#         return "\n".join(lines)


#     def build_solver_prompt(self,
#                             puzzle: Any,
#                             context: Optional[Dict[str, Any]] = None
#                             )->str:
#         base_prompt = self.prompts.get("solver_prompt",
#                                        "Solve the puzzle carefully."
#                                        )
#         context_text = self._format_context(context)
#         return f"""
#         {base_prompt}

#         Puzzle:
#         {puzzle}

#         Context:
#         {context_text}

#         Answer:
#         """.strip()


#     def build_reflection_prompt(self,
#                                 puzzle: Any,
#                                 previous_answer: Any,
#                                 evaluation_feedback: Optional[str]=None,
#                                 context: Optional[Dict[str, Any]] = None
#                                 ):
#         base_prompt = self.prompts.get("reflection_prompt",
#                                        "Analyze why the previous answer failed.")
#         context_text=self._format_context(context)
#         return f"""
#             {base_prompt}

#             Puzzle:
#             {puzzle}

#             Previous Answer:
#             {previous_answer}

#             Evaluation Feedback:
#             {evaluation_feedback or "No feedback provided."}

#             Context:
#             {context_text}

#             Reflection:
#             """.strip()


#     def build_debate_prompt(self,
#                             puzzle: Any,
#                             agent_answer: str,
#                             opponent_answer: str,
#                             )->str:
#         base_prompt = self.prompts.get("debate_prompt","Critique the other agent.")

#         return f"""
#             {base_prompt}

#             Puzzle:
#             {puzzle}

#             Agent Answer:
#             {agent_answer}

#             Opponent Answer:
#             {opponent_answer}

#             Critique:
#             """.strip()
