import copy
import random

# from dataclasses import dataclass

# @dataclass
# class Puzzle:
#     name: str


class Puzzle:
    def __init__(self, size, goal_state=None):
        self.size = size  # Puzzle dimension (n x n)
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.empty_pos = (size - 1, size - 1)  # Position of empty space

        # Initialize with a random solvable state if no goal_state provided
        if goal_state is None:
            self.randomize()
        else:
            self.grid = goal_state
            self.find_empty()

    def randomize(self, moves=100):
        """Generate a random solvable puzzle state"""
        # Start from solved state
        self.grid = [
            [i * self.size + j + 1 for j in range(self.size)] for i in range(self.size)
        ]
        self.empty_pos = (self.size - 1, self.size - 1)

        # Make random moves
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        for _ in range(moves):
            # Choose a random direction
            dx, dy = random.choice(directions)

            # Calculate new empty position
            new_x = self.empty_pos[0] + dx
            new_y = self.empty_pos[1] + dy

            # Check if move is valid
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                # Move the tile
                self.swap_tiles(self.empty_pos, (new_x, new_y))
                self.empty_pos = (new_x, new_y)

    def swap_tiles(self, pos1, pos2):
        """Swap tiles between two positions"""
        x1, y1 = pos1
        x2, y2 = pos2

        # Swap the values
        self.grid[x1][y1], self.grid[x2][y2] = self.grid[x2][y2], self.grid[x1][y1]

    def move(self, direction):
        """Move the empty space in the specified direction"""
        dx, dy = direction

        # Calculate new empty position
        new_x = self.empty_pos[0] + dx
        new_y = self.empty_pos[1] + dy

        # Check if move is valid
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            # Move the tile
            self.swap_tiles(self.empty_pos, (new_x, new_y))
            self.empty_pos = (new_x, new_y)
            return True
        return False

    def find_empty(self):
        """Find the position of the empty space"""
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    self.empty_pos = (i, j)
                    return

    def is_solved(self):
        """Check if the puzzle is in its solved state"""
        for i in range(self.size):
            for j in range(self.size):
                if i == self.size - 1 and j == self.size - 1:
                    if self.grid[i][j] != 0:
                        return False
                elif self.grid[i][j] != i * self.size + j + 1:
                    return False
        return True

    def print_puzzle(self):
        """Print the current puzzle state"""
        for i in range(self.size):
            print(" ".join(str(self.grid[i][j]).ljust(3) for j in range(self.size)))
        print()

    def get_grid(self):
        """Return the current grid state"""
        return copy.deepcopy(self.grid)

    def set_grid(self, grid):
        """Set the grid state and update empty position"""
        self.grid = grid
        self.find_empty()

    def get_empty_pos(self):
        """Return the position of the empty space"""
        return self.empty_pos

    def get_possible_moves(self):
        """Return a list of possible move directions"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        possible = []

        for dx, dy in directions:
            new_x = self.empty_pos[0] + dx
            new_y = self.empty_pos[1] + dy

            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                possible.append((dx, dy))

        return possible


# Example usage:
if __name__ == "__main__":
    # Create a 4x4 puzzle
    puzzle = Puzzle(4)

    # Print the puzzle
    puzzle.print_puzzle()

    # Move the empty space right
    puzzle.move((0, 1))

    # Print the puzzle again
    puzzle.print_puzzle()

    # Check if solved
    print("Is solved?", puzzle.is_solved())

    # Get possible moves
    print("Possible moves:", puzzle.get_possible_moves())
