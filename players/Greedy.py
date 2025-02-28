from typing import Dict, List, Tuple, Optional, Set
from pyrat import Player, Maze, GameState, Action

class Greedy(Player):
    def __init__(self, skin=None):
        """
        Initialize the Greedy player.
        I/O:
        I: skin (optional) - Appearance of the player.
        O: None
        """
        super().__init__(skin=skin)
        self.visited: Set[int] = set()
        self.path: List[int] = []
        self.current_target: Optional[int] = None
        self.cheeses: List[int] = []
        self.all_shortest_paths: Dict[int, Dict[int, List[int]]] = {}
        self.position: int = 0

    def preprocessing(self, maze: Maze, game_state: GameState) -> None:
        """
        Preprocessing phase before the game starts.
        I/O:
        I: maze - Maze object representing the game environment.
           game_state - Current state of the game (positions, cheeses, etc.).
        O: None
        - Explication: On extrait la liste des fromages, la position initiale du joueur et on calcule
          tous les chemins les plus courts entre toutes les paires de vertices pour pouvoir déterminer
          rapidement le chemin vers le fromage le plus proche.
        """
        self.cheeses = game_state.cheese.copy()
        self.position = game_state.player_locations[self.name]
        self.current_target = None
        self.all_shortest_paths = self.compute_all_pairs_shortest_paths(maze)

    def turn(self, maze: Maze, game_state: GameState) -> Action:
        """
        Decide the next action at each turn.
        I/O:
        I: maze - Maze object.
           game_state - Current state of the game (positions, remaining cheeses, etc.).
        O: Action - The next move (NORTH, SOUTH, EAST, WEST, NOTHING).
        - Explication: On vérifie si le chemin actuel est encore valide. Si le fromage ciblé
          a disparu, on recalcule la cible la plus proche. Ensuite, on avance d'un pas vers celle-ci.
        """
        self.position = game_state.player_locations[self.name]
        self.cheeses = game_state.cheese.copy()

        if not self.path or self.current_target not in self.cheeses:
            self.current_target = self.find_closest_cheese()
            if self.current_target is not None:
                self.path = self.all_shortest_paths[self.position][self.current_target].copy()
            else:
                self.path = []

        if self.path:
            next_cell = self.path.pop(0)
            action = self.get_action(self.position, next_cell, maze)
            self.position = next_cell
            return action
        else:
            return Action.NOTHING

    def compute_all_pairs_shortest_paths(self, maze: Maze) -> Dict[int, Dict[int, List[int]]]:
        """
        Compute shortest paths between all pairs of vertices.
        I/O:
        I: maze - Maze object providing the graph structure.
        O: all_paths - A dictionary where all_paths[u][v] is the shortest path from u to v.
        - Explication: Pour chaque sommet du labyrinthe, on exécute Dijkstra afin d'obtenir un tableau
          des chemins les plus courts vers tous les autres sommets, puis on stocke ces informations.
        """
        all_paths = {}
        vertices = maze.vertices
        for vertex in vertices:
            distances, previous = self.dijkstra(maze, vertex)
            paths = {}
            for target in vertices:
                paths[target] = self.find_route(previous, vertex, target)
            all_paths[vertex] = paths
        return all_paths

    def dijkstra(self, maze: Maze, source: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """
        Dijkstra's algorithm to compute shortest paths from a source vertex.
        I/O:
        I: maze - Maze object to get neighbors and weights.
           source - The starting vertex index.
        O: (distances, previous) - Distances from source to all vertices and their predecessors.
        - Explication: On initialise toutes les distances à l'infini, on utilise un ensemble de
          sommets non visités, et on met à jour les distances en choisissant le sommet le plus proche
          non visité jusqu'à ce que tous soient traités.
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
        I: previous - Dictionary of predecessors for each vertex.
           start - Starting vertex.
           target - Destination vertex.
        O: path - List of vertices representing the shortest path from start to target.
        - Explication: On remonte la chaîne des prédécesseurs à partir du target jusqu'au start
          pour reconstituer le chemin, puis on le renverse pour obtenir le sens correct.
        """
        path: List[int] = []
        current: Optional[int] = target
        while current != start:
            if current is None:
                return []
            path.insert(0, current)
            current = previous[current]
        return path

    def find_closest_cheese(self) -> Optional[int]:
        """
        Find the closest cheese from the current position using the precomputed shortest paths.
        I/O:
        I: None (use self.position, self.all_shortest_paths, self.cheeses)
        O: closest_cheese - The vertex of the closest cheese, or None if none accessible.
        - Explication: On parcourt la liste des fromages, on calcule la longueur du chemin vers
          chacun, et on choisit le fromage nécessitant le moins d'étapes.
        """
        min_distance = float('inf')
        closest_cheese = None
        for cheese in self.cheeses:
            if cheese not in self.all_shortest_paths[self.position]:
                continue
            path_length = len(self.all_shortest_paths[self.position][cheese])
            if path_length < min_distance:
                min_distance = path_length
                closest_cheese = cheese
        return closest_cheese

    def get_action(self, source: int, target: int, maze: Maze) -> Action:
        """
        Determine the action (NORTH, SOUTH, EAST, WEST) to move from source to target cell.
        I/O:
        I: source - Current vertex index.
           target - Destination vertex index.
           maze - Maze object to convert vertex indices to (row, col).
        O: Action - Direction to move towards the target (or NOTHING if no movement).
        - Explication: On convertit les indices en coordonnées (x,y), puis on regarde la différence
          entre les positions pour déterminer la direction du mouvement.
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
