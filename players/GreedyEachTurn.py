from typing import Dict, List, Tuple, Optional
from pyrat import Player, Maze, GameState, Action

class GreedyEachTurn(Player):
    def __init__(self, skin=None):
        """
        Initialize the GreedyEachTurn player.
        I/O:
        I: skin (optional) - Appearance of the player.
        O: None
        - Explication: Ce joueur recalcule le fromage cible à chaque tour,
          garantissant ainsi une adaptation instantanée aux changements
          survenus pendant le tour de l'adversaire.
        """
        super().__init__(skin=skin)
        self.position: int = 0

    def preprocessing(self, maze: Maze, game_state: GameState) -> None:
        """
        Preprocessing before the game starts.
        I/O:
        I: maze - Maze object representing the game environment.
           game_state - Current state of the game (positions, cheeses, etc.).
        O: None
        - Explication: On enregistre le labyrinthe pour pouvoir accéder à sa structure.
          Contrairement aux approches précédentes, on ne pré-calcule pas toutes les distances ici.
        """
        self.maze = maze

    def turn(self, maze: Maze, game_state: GameState) -> Action:
        """
        Determine the next action at each turn, recalculating targets each time.
        I/O:
        I: maze - Maze object.
           game_state - Current state of the game (positions, cheeses, etc.).
        O: Action - The move (NORTH, SOUTH, EAST, WEST, NOTHING).
        - Explication: À chaque tour, on identifie le fromage le plus proche,
          on calcule le chemin pour y accéder, puis on réalise un déplacement
          dans cette direction. Si un fromage disparaît, le tour suivant
          recalculera automatiquement un nouveau chemin.
        """
        self.position = game_state.player_locations[self.name]
        cheeses = game_state.cheese.copy()
        if not cheeses:
            return Action.NOTHING

        distances, previous = self.dijkstra(maze, self.position)
        closest_cheese = min(cheeses, key=lambda cheese: distances.get(cheese, float('inf')), default=None)
        if closest_cheese is None or distances[closest_cheese] == float('inf'):
            return Action.NOTHING

        path = self.find_route(previous, self.position, closest_cheese)
        if not path:
            return Action.NOTHING

        next_cell = path[0]
        action = self.get_action(self.position, next_cell, maze)
        self.position = next_cell
        return action

    def dijkstra(self, maze: Maze, source: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """
        Compute shortest paths from a source vertex using Dijkstra's algorithm.
        I/O:
        I: maze - Maze object to access neighbors and weights.
           source - Index of the starting vertex.
        O: (distances, previous) - Distances to each vertex and their predecessors.
        - Explication: On initialise les distances, puis on sélectionne à chaque itération
          le sommet non visité le plus proche, et on met à jour les distances
          des voisins jusqu'à ce que tous les sommets soient traités ou inaccessibles.
        """
        distances: Dict[int, float] = {vertex: float('inf') for vertex in maze.vertices}
        previous: Dict[int, Optional[int]] = {vertex: None for vertex in maze.vertices}
        distances[source] = 0.0
        unvisited: Set[int] = set(maze.vertices)

        while unvisited:
            current = min(unvisited, key=lambda vertex: distances[vertex])
            if distances[current] == float('inf'):
                break
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
           start - Starting vertex index.
           target - Destination vertex index.
        O: path - List of vertices forming the shortest path.
        - Explication: On remonte la chaîne de prédécesseurs à partir du target jusqu'à start,
          puis on inverse le résultat. Si on n'arrive pas jusqu'au start, le chemin est vide.
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
        Determine the direction (NORTH, SOUTH, EAST, WEST) from source to target.
        I/O:
        I: source - Current vertex index.
           target - Destination vertex index.
           maze - Maze object to translate indices into coordinates.
        O: Action - The move direction or NOTHING if no movement required.
        - Explication: On calcule les coordonnées (x,y) de source et de target.
          La différence (dx, dy) permet de déterminer la direction du mouvement.
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
