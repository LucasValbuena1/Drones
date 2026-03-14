from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP

#funciones axiliares
def restore(csp: DroneAssignmentCSP, pruned: dict[str, list]) -> None:
    for var, values in pruned.items():
        csp.domains[var].extend(values)


def forward_check(csp: DroneAssignmentCSP, stats: dict, var: str, assignment: dict) -> dict[str, list] | None:
    pruned: dict[str, list] = {}

    for neighbor in csp.get_neighbors(var):
        if neighbor in assignment:
            continue
        pruned[neighbor] = []

        for val in list(csp.domains[neighbor]):
            if not csp.is_consistent(neighbor, val, assignment):
                csp.domains[neighbor].remove(val)
                pruned[neighbor].append(val)
                stats["pruned"] += 1

        if len(csp.domains[neighbor]) == 0:
            return None
    return pruned


def select_var(csp: DroneAssignmentCSP, assignment: dict) -> str:
    unassigned = csp.get_unassigned_variables(assignment)
    return min(unassigned, key=lambda var: (
        len(csp.domains[var]),
        -sum(1 for n in csp.get_neighbors(var) if n not in assignment)
    ))


def order_values(csp: DroneAssignmentCSP, var: str, assignment: dict) -> list:
    return sorted(
        csp.domains[var],
        key=lambda value: csp.get_num_conflicts(var, value, assignment)
    )


def values_compatible(csp: DroneAssignmentCSP, xi: str, vi: str, xj: str, vj: str, assignment: dict) -> bool:
    temp = dict(assignment)
    temp[xi] = vi
    return csp.is_consistent(xj, vj, temp)


def revise(csp: DroneAssignmentCSP, xi: str, xj: str, assignment: dict) -> tuple[bool, list]:
    revised = False
    removed = []
    for vi in list(csp.domains[xi]):
        has_support = any(
            values_compatible(csp, xi, vi, xj, vj, assignment)
            for vj in csp.domains[xj]
            if xj not in assignment or assignment[xj] == vj
        )

        if not has_support:
            csp.domains[xi].remove(vi)
            removed.append(vi)
            revised = True
    return revised, removed


def ac3(csp: DroneAssignmentCSP, stats: dict, assignment: dict) -> dict[str, list] | None:
    stats["ac3_calls"] += 1
    queue = [
        (var, neighbor)
        for var in csp.variables if var not in assignment
        for neighbor in csp.get_neighbors(var)
    ]
    all_removed: dict[str, list] = {}

    while queue:
        xi, xj = queue.pop(0)
        if xi in assignment:
            continue
        revised, removed = revise(csp, xi, xj, assignment)
        if revised:
            all_removed[xi] = all_removed.get(xi, []) + removed
            if len(csp.domains[xi]) == 0:
                return None
            for xk in csp.get_neighbors(xi):
                if xk != xj and xk not in assignment:
                    queue.append((xk, xi))

    return all_removed

#------------------------------------------------------------------------------------------------
def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    stats = {"assignments": 0, "backtracks": 0}

    def backtrack(assignment: dict) -> dict | None:
        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var]:
            stats["assignments"] += 1

            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                result = backtrack(assignment)

                if result is not None:
                    return result
                csp.unassign(var, assignment)
                stats["backtracks"] += 1

        return None

    result = backtrack({})
    print(f"[backtracking] assignments: {stats['assignments']} | backtracks: {stats['backtracks']}")
    return result


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    stats = {"assignments": 0, "backtracks": 0, "pruned": 0}

    def backtrack(assignment: dict) -> dict | None:
        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in csp.domains[var]:
            stats["assignments"] += 1

            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                pruned = forward_check(csp, stats, var, assignment)

                if pruned is not None:
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                if pruned is not None:
                    restore(csp, pruned)
                csp.unassign(var, assignment)
                stats["backtracks"] += 1

        return None

    result = backtrack({})
    print(f"[FC] Assignments: {stats['assignments']} | Backtracks: {stats['backtracks']} | Pruned: {stats['pruned']}")
    return result


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    stats = {"assignments": 0, "backtracks": 0, "ac3_calls": 0}

    def backtrack(assignment: dict) -> dict | None:
        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in list(csp.domains[var]):
            stats["assignments"] += 1
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                snapshot = ac3(csp, stats, assignment)

                if snapshot is not None:
                    result = backtrack(assignment)
                    if result is not None:
                        return result

                if snapshot is not None:
                    restore(csp, snapshot)
                csp.unassign(var, assignment)
                stats["backtracks"] += 1

        return None

    initial = ac3(csp, stats, {})
    if initial is None:
        print("[AC-3] Initial AC-3 found no solution.")
        return None

    result = backtrack({})
    print(f"[AC-3] Assignments: {stats['assignments']} | Backtracks: {stats['backtracks']} | AC3 calls: {stats['ac3_calls']}")
    return result


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    stats = {"assignments": 0, "backtracks": 0, "pruned": 0}

    def backtrack(assignment: dict) -> dict | None:
        if csp.is_complete(assignment):
            return assignment

        var = select_var(csp, assignment)

        for value in order_values(csp, var, assignment):
            stats["assignments"] += 1
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)
                pruned = forward_check(csp, stats, var, assignment)

                if pruned is not None:
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                        
                if pruned is not None:
                    restore(csp, pruned)
                csp.unassign(var, assignment)
                stats["backtracks"] += 1

        return None

    result = backtrack({})
    print(f"[MRV+LCV] Assignments: {stats['assignments']} | backtracks: {stats['backtracks']} | Pruned: {stats['pruned']}")
    return result
