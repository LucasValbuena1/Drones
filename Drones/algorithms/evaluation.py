from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from world.game_state import GameState

from algorithms.utils import bfs_distance, dijkstra, manhattan_distance


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    # TODO: Implement your code here
    layout = state.get_layout()
    posicion_dron = state.get_drone_position()
    posiciones_cazadores = state.get_hunter_positions()
    entregas_pendientes = state.get_pending_deliveries()
    puntaje_actual = state.get_score()

    puntaje = puntaje_actual * 0.5 - len(entregas_pendientes) * 50

    if entregas_pendientes:
        costos = [dijkstra(layout, posicion_dron, e)[0] for e in entregas_pendientes]
        costos_validos = [c for c in costos if c is not None and c < float('inf')]
        if costos_validos:
            puntaje += 100.0 / (min(costos_validos) + 1)

    distancias_cazadores = []
    for cazador in posiciones_cazadores:
        if manhattan_distance(cazador, posicion_dron) > 10:
            distancias_cazadores.append(manhattan_distance(cazador, posicion_dron))
        else:
            d = bfs_distance(layout, cazador, posicion_dron, True)
            distancias_cazadores.append(d if d is not None and d < float('inf') else float('inf'))

    for d in distancias_cazadores:
        if d <= 5:
            puntaje -= 200.0 / (d + 0.5)

    distancias_finitas = [d for d in distancias_cazadores if d < float('inf')]
    if distancias_finitas:
        puntaje += min(min(distancias_finitas) * 10, 80)

    if entregas_pendientes and distancias_finitas:
        for entrega in entregas_pendientes:
            if manhattan_distance(posicion_dron, entrega) > 10:
                continue
            costo_dron, camino_dron = dijkstra(layout, posicion_dron, entrega)
            if costo_dron is None or costo_dron == float('inf'):
                continue
            min_cazador = min(
                (bfs_distance(layout, c, entrega, True) or float('inf'))
                for c in posiciones_cazadores
            )
            if costo_dron < min_cazador:
                puntaje += 60.0 / (costo_dron + 1)

    return max(-999.0, min(999.0, puntaje))
