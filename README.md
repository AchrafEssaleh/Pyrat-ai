# Pyrat-ai – Players

This folder contains the AI player implementations for the PyRat game:

1. **Greedy.py**
2. **GreedyEachCheese.py**
3. **GreedyEachTurn.py**
4. **DDNPlayer.py**

### Key Approaches

#### Greedy:
- Precomputes shortest paths (using a modified Dijkstra algorithm) during preprocessing.
- Chooses the closest cheese at the start and follows the precomputed path.
- If a cheese disappears mid-path, the player continues along the old route until it reaches the target or recalculates on the next decision.

#### GreedyEachCheese:
- Similar to Greedy, but recalculates the target cheese after each cheese is collected.
- More reactive: once the player reaches a cheese or if it’s taken by the opponent, a new closest cheese is chosen.
- Continues the current move if a cheese disappears mid-path until the target list is updated.

#### GreedyEachTurn:
- Recalculates the closest cheese at every turn, adapting instantly to changes in cheese availability.
- Ensures that at each decision step, the player heads toward the best current target.

#### DDNPlayer:
- Currently uses traditional heuristic techniques (including Dijkstra-based computations) to inform decision making.
- Serves as a foundation for future improvements that may incorporate deep learning techniques.
- Balances exploration and exploitation based on current maze state, with planned enhancements to integrate a neural network for more advanced decision-making.

### Why These Approaches?
- They approximate solutions for selecting which cheese to target in a TSP-like scenario.
- Each successive method improves reactivity and adaptability.
- Transitioning from Greedy to GreedyEachCheese to GreedyEachTurn, and refining DDNPlayer, enhances performance under dynamic conditions.

---

### Unit Tests and Documentation
- Unit tests verify key functions (e.g., shortest path computations, target recalculations, decision-making) using mock objects.
- Comprehensive docstrings explain the purpose, inputs, and outputs of each class and method, ensuring clarity and maintainability.

---

### Practical Considerations
1. **Complexity & Approach:**
   - Each heuristic uses shortest path computations, generally O((V + E) log V).
   - More frequent recalculations improve adaptability at the cost of increased computation.

2. **Dependencies:**
   - Relies on standard Python libraries and the PyRat framework.

3. **Future Improvements:**
   - Future work will focus on refining DDNPlayer by incorporating deep learning techniques and possibly integrating opponent prediction models for further performance gains.
