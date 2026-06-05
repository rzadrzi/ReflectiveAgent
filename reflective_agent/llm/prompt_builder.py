from typing import Any, Dict, Optional

class PromptBuilder:
    def __init__(self, prompts: Dict[str, str]) :
        self.prompts = prompts

    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        if context is None:
            return "No context provided."
        lines = []

        for key, value in context.items():
            lines.append(f"-{key}: {value}")

        return "\n".join(lines)


    def build_solver_prompt(self,
                            puzzle: Any,
                            context: Optional[Dict[str, Any]] = None
                            )->str:
        base_prompt = self.prompts.get("solver_prompt",
                                       "Solve the puzzle carefully."
                                       )
        context_text = self._format_context(context)
        return f"""
        {base_prompt}
        
        Puzzle: 
        {puzzle}
        
        Context: 
        {context_text}
        
        Answer:
        """.strip()


    def build_reflection_prompt(self,
                                puzzle: Any,
                                previous_answer: Any,
                                evaluation_feedback: Optional[str]=None,
                                context: Optional[Dict[str, Any]] = None
                                ):
        base_prompt = self.prompts.get("reflection_prompt",
                                       "Analyze why the previous answer failed.")
        context_text=self._format_context(context)
        return f"""
            {base_prompt}
            
            Puzzle: 
            {puzzle}
            
            Previous Answer:
            {previous_answer}
            
            Evaluation Feedback:
            {evaluation_feedback or "No feedback provided."}
            
            Context:
            {context_text}
            
            Reflection:
            """.strip()

    
    def build_debate_prompt(self,
                            puzzle: Any,
                            agent_answer: str,
                            opponent_answer: str,
                            )->str:
        base_prompt = self.prompts.get("debate_prompt","Critique the other agent.")

        return f"""
            {base_prompt}
            
            Puzzle: 
            {puzzle}
            
            Agent Answer:
            {agent_answer}
            
            Opponent Answer:
            {opponent_answer}
            
            Critique:
            """.strip()