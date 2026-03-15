from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        # TODO: Implement your code here
        numero_agentes = state.get_num_agents()
        acciones_legales_dron = state.get_legal_actions(0)

        mejor_accion = None
        mejor_puntaje = float('-inf')
        
        def minimax(estado_actual, profundidad_restante, indice_agente_actual):
                # Caso terminal: victoria o derrota
                if estado_actual.is_win() or estado_actual.is_lose():
                    return self.evaluation_function(estado_actual)

                # Todos los agentes movieron: empieza nuevo ply
                if indice_agente_actual == numero_agentes:
                    indice_agente_actual = 0
                    profundidad_restante -= 1

                # Profundidad agotada: evaluar con heurística
                if profundidad_restante == 0:
                    return self.evaluation_function(estado_actual)

                acciones_legales_agente = estado_actual.get_legal_actions(indice_agente_actual)
                if not acciones_legales_agente:
                    return self.evaluation_function(estado_actual)

                estados_sucesores = [
                    estado_actual.generate_successor(indice_agente_actual, accion)
                    for accion in acciones_legales_agente]

                # Nodo MAX: turno del dron (agente 0)
                if indice_agente_actual == 0:
                    return max(
                        minimax(sucesor, profundidad_restante, indice_agente_actual + 1)
                        for sucesor in estados_sucesores)

                # Nodo MIN: turno de un cazador (agentes 1..N)
                else:
                    return min(
                        minimax(sucesor, profundidad_restante, indice_agente_actual + 1)
                        for sucesor in estados_sucesores)

                for accion_candidata in acciones_legales_dron:
                    
                    estado_sucesor_dron = state.generate_successor(0, accion_candidata)
                    puntaje_accion = minimax(estado_sucesor_dron, self.depth, 1)
                    if puntaje_accion > mejor_puntaje:
                        mejor_puntaje = puntaje_accion
                        mejor_accion = accion_candidata

                return mejor_accion


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        # TODO: Implement your code here (BONUS)
        numero_agentes = state.get_num_agents()
        acciones_legales_dron = state.get_legal_actions(0)

        mejor_accion = None
        mejor_puntaje = float('-inf')
        alfa = float('-inf')
        beta = float('inf')

        def alpha_beta(estado_actual, profundidad_restante, indice_agente_actual, alfa, beta):
            # Caso terminal: victoria o derrota
            if estado_actual.is_win() or estado_actual.is_lose():
                return self.evaluation_function(estado_actual)

            # Todos los agentes movieron: empieza nuevo ply
            if indice_agente_actual == numero_agentes:
                indice_agente_actual = 0
                profundidad_restante -= 1

            # Profundidad agotada: evaluar con heurística
            if profundidad_restante == 0:
                return self.evaluation_function(estado_actual)

            acciones_legales_agente = estado_actual.get_legal_actions(indice_agente_actual)
            if not acciones_legales_agente:
                return self.evaluation_function(estado_actual)

            # Nodo MAX: turno del dron (agente 0)
            if indice_agente_actual == 0:
                mejor_valor = float('-inf')
                for accion in acciones_legales_agente:
                    estado_sucesor = estado_actual.generate_successor(indice_agente_actual, accion)
                    valor_sucesor = alpha_beta(estado_sucesor, profundidad_restante, indice_agente_actual + 1, alfa, beta)
                    mejor_valor = max(mejor_valor, valor_sucesor)
                    alfa = max(alfa, mejor_valor)
                    # Poda: el nodo MIN padre nunca elegirá este camino
                    if mejor_valor > beta:
                        break
                return mejor_valor

            # Nodo MIN: turno de un cazador (agentes 1..N)
            else:
                mejor_valor = float('inf')
                for accion in acciones_legales_agente:
                    estado_sucesor = estado_actual.generate_successor(indice_agente_actual, accion)
                    valor_sucesor = alpha_beta(estado_sucesor, profundidad_restante, indice_agente_actual + 1, alfa, beta)
                    mejor_valor = min(mejor_valor, valor_sucesor)
                    beta = min(beta, mejor_valor)
                    # Poda: el nodo MAX padre nunca elegirá este camino
                    if mejor_valor < alfa:
                        break
                return mejor_valor

        for accion_candidata in acciones_legales_dron:
            estado_sucesor_dron = state.generate_successor(0, accion_candidata)
            puntaje_accion = alpha_beta(estado_sucesor_dron, self.depth, 1, alfa, beta)
            if puntaje_accion > mejor_puntaje:
                mejor_puntaje = puntaje_accion
                mejor_accion = accion_candidata
                alfa = max(alfa, mejor_puntaje)

        return mejor_accion




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
        numero_agentes = state.get_num_agents()
        acciones_legales_dron = state.get_legal_actions(0)

        mejor_accion = None
        mejor_puntaje = float('-inf')

        def expectimax(estado_actual, profundidad_restante, indice_agente_actual):
            if estado_actual.is_win() or estado_actual.is_lose():
                return self.evaluation_function(estado_actual)

            if indice_agente_actual == numero_agentes:
                indice_agente_actual = 0
                profundidad_restante -= 1

            if profundidad_restante == 0:
                return self.evaluation_function(estado_actual)

            acciones_legales_agente = estado_actual.get_legal_actions(indice_agente_actual)
            if not acciones_legales_agente:
                return self.evaluation_function(estado_actual)

            estados_sucesores = [
                estado_actual.generate_successor(indice_agente_actual, accion)
                for accion in acciones_legales_agente
            ]
            
            if indice_agente_actual == 0:
                return max(
                    expectimax(sucesor, profundidad_restante, indice_agente_actual + 1)
                    for sucesor in estados_sucesores
                )

            else:
                valores_hijos = [
                    expectimax(sucesor, profundidad_restante, indice_agente_actual + 1)
                    for sucesor in estados_sucesores
                ]
                valor_greedy = min(valores_hijos)
                valor_aleatorio = sum(valores_hijos) / len(valores_hijos)
                return (1 - self.prob) * valor_greedy + self.prob * valor_aleatorio

        for accion_candidata in acciones_legales_dron:
            estado_sucesor_dron = state.generate_successor(0, accion_candidata)
            puntaje_accion = expectimax(estado_sucesor_dron, self.depth, 1)
            if puntaje_accion > mejor_puntaje:
                mejor_puntaje = puntaje_accion
                mejor_accion = accion_candidata

        return mejor_accion
