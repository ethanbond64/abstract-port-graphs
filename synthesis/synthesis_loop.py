from typing import List

from arc.arc_objects import ArcGraph
from arc.arc_utils import ArcTask
from synthesis.goals import GoalGraph, RootGoal
from synthesis.meta_learning import MetaModel, ActionLogger
from synthesis.program_state import ProgramState
from synthesis.arc_specific.temp_graph_finisher_logic import draft_program_to_arc_graph


def synthesis_loop(meta_model: MetaModel,
                   action_logger: ActionLogger,
                   challenge: ArcTask) -> ArcGraph:

    root_goal = RootGoal()
    goal_graph = GoalGraph(root_goal)
    program_state = ProgramState(challenge)

    iteration = 0

    while not root_goal.accomplished(program_state):

        if iteration % 50 == 0:
            print(f"Iteration {iteration}", len(program_state.get_trace_graph().get_all_traces()))

        active_goals = goal_graph.get_active_goals()
        chosen_goal, chosen_action = meta_model.decide_action(program_state, active_goals)

        if chosen_action:

            # Action is expected to update the program state internally.
            result = chosen_action.run(program_state)
            chosen_goal.record_action_result(chosen_action, result)

            # Log for retroactive learning
            action_logger.log_action(chosen_goal, chosen_action, result)

        goal_graph.refresh(program_state, chosen_goal)
        iteration += 1

    return draft_program_to_arc_graph(root_goal.draft_program)
