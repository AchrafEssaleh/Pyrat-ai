from typing import Dict, List, Tuple, Optional, Set
from pyrat import Player, Maze, GameState, Action

class GreedyEachCheese(Player):
    def __init__(self, skin=None):
        """
        Initialize the GreedyEachCheese player.
        I/O:
        I: skin (optional) - Appearance of the player.
        O: None
        """
        super().__init__(skin=skin)
        self.position: int = 0

    def preprocessing(self, maze: Maze, game_state: GameState) -> None:
        """
        Preprocessing phase executed before the game starts.
        I/O:
        I: maze - Maze object representing the game environment.
           game_state - Current state of the game (positions, cheeses, etc.).
        O: None
        - Explication: Cette phase permet d'enregistrer une référence au labyrinthe.
          Contrairement à la version initiale (Greedy), ici on ne pré-calcule pas tous les
          chemins, car on recalcule à chaque fromage atteint.
        """
        self.maze = maze

    def turn(self, maze: Maze, game_state: GameState) -> Action:
        """
        Decide the next action at each turn, recalculating after each cheese is grabbed.
        I/O:
        I: maze - Maze object.
           game_state - Current state of the game (positions, remaining cheeses, etc.).
        O: Action - The next move (NORTH, SOUTH, EAST, WEST, NOTHING).
        - Explication: À chaque tour, on identifie le fromage le plus proche à partir
          de la position actuelle. Une fois déterminé, on calcule le chemin pour y arriver
          et on effectue un mouvement dans sa direction. Si le fromage ciblé disparaît entre-temps,
          on recalculera au tour suivant.
        """
        self.position = game_state.player_locations[self.name]
        cheeses = game_state.cheese.copy()
        if not cheeses:
            return Action.NOTHING

        distances, previous = self.dijkstra(maze, self.position)
        closest_cheese = min(cheeses, key=lambda cheese: distances.get(cheese, float('inf')))
        path = self.find_route(previous, self.position, closest_cheese)
        if not path:
            return Action.NOTHING

        next_cell = path[0]
        action = self.get_action(self.position, next_cell, maze)
        self.position = next_cell
        return action

    def dijkstra(self, maze: Maze, source: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """
        Dijkstra's algorithm to compute shortest paths from a source vertex.
        I/O:
        I: maze - Maze object to get neighbors and weights.
           source - The starting vertex index.
        O: (distances, previous) - Distances from source to all vertices and their predecessors.
        - Explication: On initialise toutes les distances à l'infini, on traite chaque
          sommet en choisissant le plus proche non visité, puis on met à jour les distances
          et précédents pour aboutir au plus court chemin jusqu'à chaque autre vertex.
        """
        distances: Dict[int, float] = {vertex: float('inf') for vertex in maze.vertices}
        previous: Dict[int, Optional[int]] = {vertex: None for vertex in maze.vertices}
        distances[source] = 0.0
        unvisited: Set[int] = set(maze.vertices)

        while unvisited:
            current = min(unvisited, key=lambda vertex: distances[vertex])
            unvisited.remove(current)

            for neighbor in maze.get_neighbors(current):
                weight = maze.get_weight(current, neighbor)
                alt = distances[current] + weight
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = current

        return distances, previous

    def find_route(self, previous: Dict[int, Optional[int]], start: int, target: int) -> List[int]:
        """
        Reconstruct the shortest path from start to target.
        I/O:
        I: previous - Dictionary mapping each vertex to its predecessor.
           start - Starting vertex.
           target - Destination vertex.
        O: path - List of vertices forming the shortest path.
        - Explication: On suit les prédécesseurs depuis la cible jusqu'au sommet de départ,
          puis on inverse le chemin obtenu pour l'avoir dans le bon sens.
        """
        path: List[int] = []
        current: Optional[int] = target
        while current != start:
            if current is None:
                return []
            path.insert(0, current)
            current = previous[current]
        return path

    def get_action(self, source: int, target: int, maze: Maze) -> Action:
        """
        Determine the action (NORTH, SOUTH, EAST, WEST) to move from source to target cell.
        I/O:
        I: source - Current vertex index.
           target - Destination vertex index.
           maze - Maze object to translate vertex indices to coordinates.
        O: Action - Movement direction towards the target cell or NOTHING if no movement.
        - Explication: On convertit indices source et cible en coordonnées (x,y),
          puis on déduit la direction en fonction de la différence entre (x,y) source et cible.
        """
        x1, y1 = maze.i_to_rc(source)
        x2, y2 = maze.i_to_rc(target)
        dx = x2 - x1
        dy = y2 - y1
        if dx == -1:
            return Action.NORTH
        elif dx == 1:
            return Action.SOUTH
        elif dy == -1:
            return Action.WEST
        elif dy == 1:
            return Action.EAST
        else:
            return Action.NOTHING
