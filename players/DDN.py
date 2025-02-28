#ESSALEH Achraf  && _El Bari Sara && _ Hanin gharsalli
from typing import Dict, List, Tuple, Optional
from pyrat import Player, Maze, GameState, Action
import math
import heapq
#Density_Driven_Navigator
class DDN(Player):
    def __init__(self, skin=None):
        super().__init__(skin=skin)
        self.maze: Optional[Maze] = None
        self.adversary_name: Optional[str] = None
        self.cheeses: List[int] = []
        self.vertices: List[int] = []

    def preprocessing(self, maze: Maze, game_state: GameState) -> None:
        self.maze = maze
        self.vertices = maze.vertices
        # Identifier le nom de l'adversaire
        for pname in game_state.player_locations:
            if pname != self.name:
                self.adversary_name = pname
                break
        self.cheeses = game_state.cheese.copy()

    def turn(self, maze: Maze, game_state: GameState) -> Action:
        if not self.maze:
            return Action.NOTHING

        # Mettre à jour les positions
        self.position = game_state.player_locations[self.name]
        adversary_pos = game_state.player_locations.get(self.adversary_name, self.position)
        self.cheeses = game_state.cheese.copy()

        if not self.cheeses:
            return Action.NOTHING

        # Calculer les distances depuis le rat et l'adversaire
        distances_rat, previous_rat = self.dijkstra(maze, self.position)
        distances_adv, _ = self.dijkstra(maze, adversary_pos)

        # Assigner les fromages en fonction des distances
        assigned_cheeses = self.assign_cheeses(distances_rat, distances_adv, self.cheeses)

        if not assigned_cheeses:
            # Fallback: choisir le fromage le plus proche accessible
            closest_cheese = self.get_closest_accessible_cheese(distances_rat, self.cheeses)
            if closest_cheese is not None:
                path = self.find_route(previous_rat, self.position, closest_cheese)
                if path:
                    return self.get_action(self.position, path[0], maze)
            # Si aucun fromage accessible, explorer
            return self.move_towards_density(maze, distances_rat)

        # Choisir le fromage le plus prioritaire assigné
        target_cheese = assigned_cheeses[0][0]
        path = self.find_route(previous_rat, self.position, target_cheese)
        if path:
            return self.get_action(self.position, path[0], maze)

        # Si le chemin est bloqué, explorer
        return self.move_towards_density(maze, distances_rat)

    def dijkstra(self, maze: Maze, source: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        distances: Dict[int, float] = {vertex: math.inf for vertex in maze.vertices}
        previous: Dict[int, Optional[int]] = {vertex: None for vertex in maze.vertices}
        distances[source] = 0.0
        priority_queue = [(0.0, source)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex]:
                continue

            for neighbor in maze.get_neighbors(current_vertex):
                weight = maze.get_weight(current_vertex, neighbor)
                alt = current_distance + weight
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (alt, neighbor))

        return distances, previous

    def find_route(self, previous: Dict[int, Optional[int]], start: int, target: int) -> List[int]:
        path: List[int] = []
        current: Optional[int] = target
        while current != start:
            if current is None:
                return []
            path.insert(0, current)
            current = previous.get(current, None)
        return path

    def assign_cheeses(self, distances_rat: Dict[int, float], distances_adv: Dict[int, float], cheeses: List[int]) -> List[Tuple[int, float]]:
        # Créer une liste de fromages où le rat peut arriver avant l'adversaire
        available_cheeses = []
        for cheese in cheeses:
            d_rat = distances_rat.get(cheese, math.inf)
            d_adv = distances_adv.get(cheese, math.inf)
            if d_rat < d_adv:
                available_cheeses.append((cheese, d_rat))
            elif d_rat == d_adv:
                available_cheeses.append((cheese, d_rat))
        # Trier les fromages par distance croissante
        available_cheeses.sort(key=lambda x: x[1])
        # Vérifier les conflits
        final_assigned = []
        targeted_cheeses = set()
        for cheese, d_rat in available_cheeses:
            if cheese in targeted_cheeses:
                continue
            if distances_adv.get(cheese, math.inf) < distances_rat.get(cheese, math.inf):
                continue
            final_assigned.append((cheese, d_rat))
            targeted_cheeses.add(cheese)
        return final_assigned

    def get_closest_accessible_cheese(self, distances: Dict[int, float], cheeses: List[int]) -> Optional[int]:
        closest_cheese = None
        closest_dist = math.inf
        for cheese in cheeses:
            d = distances.get(cheese, math.inf)
            if d < closest_dist:
                closest_dist = d
                closest_cheese = cheese
        if closest_dist < math.inf:
            return closest_cheese
        return None

    def move_towards_density(self, maze: Maze, distances_rat: Dict[int, float]) -> Action:
        # Calculer la densité de fromages autour de chaque cellule
        density: Dict[int, float] = {}
        for cell in maze.vertices:
            density[cell] = sum(
                1.0 / (1.0 + distances_rat.get(ch, math.inf))
                for ch in self.cheeses
                if distances_rat.get(ch, math.inf) != math.inf
            )

        current_density = density.get(self.position, 0.0)
        best_action = Action.NOTHING
        best_density = current_density

        # Trouver l'action qui mène à la cellule avec la densité la plus élevée
        for action in [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]:
            new_pos = self.simulate_action(self.position, action, maze)
            if new_pos is None:
                continue
            new_density = density.get(new_pos, 0.0)
            if new_density > best_density:
                best_density = new_density
                best_action = action

        # Si aucune action n'augmente la densité, choisir une action valide aléatoire
        if best_action == Action.NOTHING:
            valid_moves = self.get_possible_actions(maze, self.position)
            if valid_moves:
                best_action = valid_moves[0]

        return best_action

    def get_action(self, source: int, target: int, maze: Maze) -> Action:
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

    def simulate_action(self, pos: int, action: Action, maze: Maze) -> Optional[int]:
        x, y = maze.i_to_rc(pos)
        if action == Action.NORTH:
            x -= 1
        elif action == Action.SOUTH:
            x += 1
        elif action == Action.EAST:
            y += 1
        elif action == Action.WEST:
            y -= 1
        else:
            return pos
        return maze.rc_to_i(x, y)

    def get_possible_actions(self, maze: Maze, position: int) -> List[Action]:
        actions = [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST]
        valid_moves = []
        x, y = maze.i_to_rc(position)
        for a in actions:
            nx, ny = x, y
            if a == Action.NORTH:
                nx -= 1
            elif a == Action.SOUTH:
                nx += 1
            elif a == Action.EAST:
                ny += 1
            elif a == Action.WEST:
                ny -= 1
            if maze.rc_to_i(nx, ny) is not None:
                valid_moves.append(a)
        return valid_moves