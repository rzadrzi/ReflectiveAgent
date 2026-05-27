

class ReflectiveAgent:
    def __init__(self, solver, evaluator, reflector, memory, config):
        self.solver = solver
        self.evaluator = evaluator
        self.reflector = reflector
        self.memory = memory
        self.config = config

    def solve(self, puzzle):
        attempts = 0
        reflections = 0
        context = {}

        while attempts < self.config.max_attempts:
            result = self.solver.solve(puzzle, context=context)
            success = self.evaluator.evaluate(puzzle, result)

            if success:
                self.memory.store_success(puzzle, result, context)
                return {
                    "answer": result,
                    "success": True,
                    "attempts": attempts + 1,
                    "agent_type": "reflective"
                }

            if not self.config.reflection_enabled or reflections >= self.config.max_reflections:
                break

            reflection = self.reflector.analyze(puzzle, result)
            self.memory.store_failure(puzzle, result, reflection)

            context = self.reflector.update_context(context, reflection)
            reflections += 1
            attempts += 1

        return {
            "answer": result,
            "success": False,
            "attempts": attempts + 1,
            "agent_type": "reflective"
        }
