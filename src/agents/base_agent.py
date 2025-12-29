# Agent Base
# the rule is Orchsteration


class BaseAgent:
    def __init__(self, solver, evaluator):
        self.solver = solver
        self.evaluator = evaluator

    def solve(self, puzzle):
        result = self.solver.solve(puzzle)
        success = self.evaluator.evaluate(puzzle, result)

        return {"answer": result, "success": success, "agent_type": "base"}
