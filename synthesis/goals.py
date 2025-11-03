import abc
import random
from collections import defaultdict
from copy import copy
from typing import List, Any, Optional, Tuple, Dict, DefaultDict, Set

from arc.arc_objects import ArcObject, EnvironmentShape
from synthesis.design_time_models import WildCardNode
from synthesis.program_state import PerceptionModel, ProgramTrace
from synthesis.execution import Action, PerceptionAction, SynthesisAction, ActionType
from synthesis.program_state import ProgramState, EvaluationContext
from synthesis.synthesis_engine import SynthesisDirective
from synthesis.reevaluation import reevaluate_trace
from synthesis.synthesis_models import DirectiveType
from synthesis.synthesis_action_selection import ProgramHypothesis, \
    predicate_directive_selection, predicate_complete_condition, operator_complete_condition, bootstrap_trace


class ReevaluationAction(Action):

    def __init__(self, trace: ProgramTrace, evaluation_context: EvaluationContext):
        self.trace = trace
        self.evaluation_context = evaluation_context

    def get_action_type(self):
        return ActionType.SYNTHESIS

    def run(self, program_state: ProgramState) -> Any:
        return reevaluate_trace(program_state, self.trace, self.evaluation_context)


class Goal(abc.ABC):

    __global_id = 0

    @staticmethod
    def __get_next_id():
        Goal.__global_id += 1
        return Goal.__global_id

    def __init__(self, parent: Optional['Goal'] = None):
        self.id = Goal.__get_next_id()
        self.__parent_id: Optional[int] = parent.id if parent else None
        self.__children: List[Goal] = []
        self.__abandoned = False

        self._action_history: List[Tuple[Action, Any]] = []


    def get_parent_id(self) -> Optional[int]:
        return self.__parent_id

    @abc.abstractmethod
    def accomplished(self, program_state: ProgramState) -> bool:
        return False

    @abc.abstractmethod
    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def record_action_result(self, action: Action, result: Any):
        self._action_history.append((action, result))

    @abc.abstractmethod
    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        ...

    @abc.abstractmethod
    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        ...

    @abc.abstractmethod
    def get_new_subgoals(self, program_state: ProgramState) -> List['Goal']:
        return []

    @abc.abstractmethod
    def get_actions(self) -> List[Action]:
        return []

    def get_action_history(self) -> List[Tuple[Action, Any]]:
        return self._action_history

class RootGoal(Goal):

    def __init__(self):
        self.perception_in = None
        self.perception_out = None
        self.perception_found = False
        self.program_complete = False
        self.draft_program = None
        super().__init__()

    def accomplished(self, program_state: ProgramState) -> bool:
        return self.program_complete

    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        if isinstance(subgoal, PerceptionGoal):
            self.perception_found = True
            self.perception_in = subgoal.current_perception
            self.perception_out = subgoal.current_perception
        elif isinstance(subgoal, SynthesisGoal):
            self.program_complete = True
            self.draft_program = subgoal.draft_program
        else:
            raise NotImplementedError()

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        raise NotImplementedError("Oof")

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:
        subgoals = []

        if not self.perception_found:
            subgoals.append(PerceptionGoal(parent=self))
        elif not self.program_complete:
            subgoals.append(SynthesisGoal(
                foundational_input_perception_model=self.perception_in,
                foundational_output_perception_model=self.perception_out,
                parent=self
            ))

        return subgoals

    def get_actions(self) -> List[Action]:
        return []


class PerceptionGoal(Goal):

    def __init__(self, parent: Goal):
        self.attempted_perceptions = []
        self.current_perception = None
        super().__init__(parent)

    def accomplished(self, program_state: ProgramState) -> bool:
        return len(program_state.get_perception_xref().get_all_perception_models()) > 0 and self.current_perception is not None

    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        pass

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        pass

    def record_action_result(self, action: Action, result: Any):
        if isinstance(action, PerceptionAction) and result:
            self.current_perception = result[0] # TODO Strongly type this
        super().record_action_result(action, result)

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:
        return []

    def get_actions(self) -> List[Action]:
        return [PerceptionAction()]


class OperatorSynthesisExploreGoal(Goal):

    def __init__(self, parent: Goal,
                 evaluation_context: EvaluationContext,
                 root_value_type: type,
                 root_trace: ProgramTrace,
                 focus_node: WildCardNode,
                 perceived_subset: Optional[List[int]] = None,
                 allowed_delta_depth: int = 3):
        self.evaluation_context = copy(evaluation_context)
        self.root_value_type = root_value_type
        self.root_trace = root_trace
        self.focus_node = focus_node
        self.perceived_subset = perceived_subset
        self.max_depth = root_trace.depth + allowed_delta_depth
        self.generated_traces: List[ProgramTrace] = [root_trace]
        self.directive_cache: Set[Tuple[Tuple[int, ...], DirectiveType]] = set()
        self.actions = []
        self.evaluation_context.perceived_subset = self.perceived_subset
        super().__init__(parent)

    def record_action_result(self, action: Action, result: Any):
        super().record_action_result(action, result)
        if result is not None:
            for directive, resulting_traces in result:

                # Record resulting traces
                self.generated_traces.extend(resulting_traces)

    def accomplished(self, program_state: ProgramState) -> bool:
        self.__enqueue_actions(program_state)
        complete = len(self.actions) == 0
        # if complete:
        #     print("EXPLORED", len(self.generated_traces))
        return complete

    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def __enqueue_actions(self, program_state: ProgramState):
        pass
        # directives: List[SynthesisDirective] = generate_all_directives(program_state,
        #                                                                self.evaluation_context,
        #                                                                self.root_value_type,
        #                                                                self.generated_traces,
        #                                                                self.max_depth)
        # self.actions = list(map(lambda d: SynthesisAction([d], self.evaluation_context), directives))

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        pass

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        pass

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:
        return []

    def get_actions(self) -> List[Action]:
        actions_local = self.actions
        self.actions = []
        return actions_local


class OperatorSynthesisValidateGoal(Goal):

    def __init__(self, parent: Goal,
                 evaluation_context: EvaluationContext,
                 traces_to_validate: List[ProgramTrace],
                 perceived_subset: Optional[Set[int]] = None):
        self.evaluation_context = copy(evaluation_context)
        self.traces_to_validate = traces_to_validate
        self.perceived_subset = perceived_subset
        self.evaluated_programs: List[ProgramTrace] = []
        self.index = 0
        super().__init__(parent)

    def accomplished(self, program_state: ProgramState) -> bool:
        return self.index >= len(self.traces_to_validate)

    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        pass

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        pass

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:
        return []

    def get_actions(self) -> List[Action]:
        if self.index >= len(self.traces_to_validate):
            return []

        if self.perceived_subset is not None:
            self.evaluation_context.perceived_subset = set(self.perceived_subset)

        trace = self.traces_to_validate[self.index]
        self.index += 1
        self.evaluated_programs.append(trace)

        action = ReevaluationAction(trace, self.evaluation_context)
        return [action]


class OperatorOrchestrateSynthesisGoal(Goal):

    def __init__(self, parent: Goal,
                 draft_program: ProgramHypothesis,
                 evaluation_context: EvaluationContext,
                 root_value_type: type):
        self.draft_program = draft_program
        self.evaluation_context = copy(evaluation_context)
        self.root_value_type = root_value_type
        self.bootstrapped = False
        self.root_trace: Optional[ProgramTrace] = None
        self.initial_programs_with_posterior: List[ProgramTrace] = []
        self.used_root_traces: Set[int] = set()
        self.subgoal_phase = "initial"  # "initial", "explore", "validate"
        self.alternation_count = 0
        self.current_explore_traces: List[ProgramTrace] = []
        self.abandoned_explore_count = 0

        self.failed = False

        super().__init__(parent)

    def accomplished(self, program_state: ProgramState) -> bool:
        is_complete = operator_complete_condition(program_state, self.evaluation_context,
                                                  self.draft_program, self.root_value_type)
        return is_complete or self.failed #or len(self._action_history) > 50

    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        if self.subgoal_phase == "initial" and isinstance(subgoal, OperatorSynthesisExploreGoal):
            # Store programs with eval_posterior > 0 from initial exploration
            for trace in subgoal.generated_traces:
                eval_result = program_state.get_evaluation_matrix().get_evaluation(trace.id, self.evaluation_context)
                if eval_result.eval_posterior > 0:
                    self.initial_programs_with_posterior.append(trace)
            self.subgoal_phase = "explore"

        elif self.subgoal_phase == "explore" and isinstance(subgoal, OperatorSynthesisExploreGoal):
            # Check if we found a successful program or hit action limit
            success = False
            programs_to_validate = []
            print("discoveries", len(subgoal.generated_traces))
            for trace in subgoal.generated_traces:
                if trace.wildcard_count == 0:
                    print("GOOD WORK")
                    eval_result = program_state.get_evaluation_matrix().get_evaluation(trace.id, self.evaluation_context)
                    if all(p_id in eval_result.instance_ids for p_id in subgoal.perceived_subset):
                        programs_to_validate.append(trace)
                        success = True

            if not success or len(subgoal._action_history) >= 75:
                self.abandoned_explore_count += 1
            else:
                self.current_explore_traces = programs_to_validate
                self.subgoal_phase = "validate"

        elif self.subgoal_phase == "validate" and isinstance(subgoal, OperatorSynthesisValidateGoal):
            self.subgoal_phase = "explore"
            self.alternation_count += 1

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        # Unimplemented
        pass

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:

        if self.accomplished(program_state):
            return []

        subgoals = []

        if not self.bootstrapped:
            self.root_trace = bootstrap_trace(program_state, self.evaluation_context, self.root_value_type, False)
            self.bootstrapped = True

        if self.subgoal_phase == "initial":
            print("Sub init")
            # Initial exploration with bootstrapped trace
            subgoals.append(OperatorSynthesisExploreGoal(
                parent=self,
                evaluation_context=self.evaluation_context,
                root_value_type=self.root_value_type,
                root_trace=self.root_trace,
                focus_node=self.root_trace.graph.get_node_by_id(self.root_trace.focus_node_id),
                perceived_subset=None,
                allowed_delta_depth=2
            ))

        elif self.subgoal_phase == "explore":
            print("Sub explore")
            # Find best unused trace from initial programs
            best_trace = None
            best_posterior = -1

            for trace in self.initial_programs_with_posterior:
                if trace.depth > 0: # TODO hardcoded so we only look at leaves
                    if trace.id not in self.used_root_traces:
                        eval_result = program_state.get_evaluation_matrix().get_evaluation(trace.id, self.evaluation_context)
                        if eval_result.eval_posterior > best_posterior:
                            best_posterior = eval_result.eval_posterior
                            best_trace = trace

            if best_trace is not None:
                print("focus trace", best_trace.id)
                self.used_root_traces.add(best_trace.id)

                # Get perceived output ids that successfully evaluated
                eval_result = program_state.get_evaluation_matrix().get_evaluation(best_trace.id, self.evaluation_context)
                instances_dict = eval_result.get_instances(program_state, None)

                successful_ids_by_case: DefaultDict[int, List[int]] = defaultdict(list)
                for perceived_id in instances_dict.keys():
                    perceived_value = program_state.get_perception_xref().get_perceived_value(perceived_id)
                    if perceived_value.output:
                        case_index = perceived_value.case_index
                        successful_ids_by_case[case_index].append(perceived_id)

                # Pick two from different cases
                perceived_subset = set()
                cases_with_ids = list(successful_ids_by_case.keys())
                if len(cases_with_ids) >= 2:
                    # Pick first id from first two cases
                    random_case_1 = random.choice(cases_with_ids)
                    random_case_2 = random.choice(cases_with_ids)
                    while random_case_2 == random_case_1:
                        random_case_2 = random.choice(cases_with_ids)

                    random_id_1 = random.choice(successful_ids_by_case[random_case_1])
                    random_id_2 = random.choice(successful_ids_by_case[random_case_2])

                    perceived_subset.add(random_id_1)
                    perceived_subset.add(random_id_2)
                elif len(cases_with_ids) == 1:
                    if len(successful_ids_by_case[cases_with_ids[0]]) >= 2:
                        perceived_subset.update(successful_ids_by_case[cases_with_ids[0]][:2])

                if len(perceived_subset) > 0:
                    subgoals.append(OperatorSynthesisExploreGoal(
                        parent=self,
                        evaluation_context=self.evaluation_context,
                        root_value_type=self.root_value_type,
                        root_trace=best_trace,
                        focus_node=best_trace.graph.get_node_by_id(best_trace.focus_node_id),
                        perceived_subset=list(perceived_subset),
                        allowed_delta_depth=4
                    ))

            else:
                self.failed = True
        elif self.subgoal_phase == "validate":
            print("Sub validate")
            # Validate traces from most recent explore
            subgoals.append(OperatorSynthesisValidateGoal(
                parent=self,
                evaluation_context=self.evaluation_context,
                traces_to_validate=self.current_explore_traces,
                perceived_subset=None
            ))

        return subgoals

    def get_actions(self) -> List[Action]:
        return []


class OperatorSynthesisGoal(Goal):

    def __init__(self, parent: Goal,
                 draft_program: ProgramHypothesis,
                 evaluation_context: EvaluationContext,
                 root_value_type: type):
        # TODO orchestrate subgoals
        self.draft_program = draft_program
        self.evaluation_context = copy(evaluation_context)
        self.root_value_type = root_value_type
        self.bootstrapped = False
        self.root_trace: Optional[ProgramTrace] = None
        self.evaluation_context.predicate_mode = False

        # TODO remove once moved to orchestration
        self.actions: List[Action] = []
        super().__init__(parent)

    def accomplished(self, program_state: ProgramState) -> bool:
        is_complete = operator_complete_condition(program_state, self.evaluation_context,
                                                  self.draft_program, self.root_value_type)
        if not is_complete and not self.bootstrapped:
            self.root_trace = bootstrap_trace(program_state, self.evaluation_context, self.root_value_type, False)
            self.bootstrapped = True

        # TODO remove once orchestrating
        if not is_complete:
            self.__enqueue_actions(program_state)

        return is_complete

    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        # TODO orchestrate subgoals
        pass

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        pass

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:
        # TODO orchestrate subgoals
        return []

    def __enqueue_actions(self, program_state: ProgramState):
        pass
        # # TODO orchestrate subgoals
        # directives: List[SynthesisDirective] = operator_directive_selection(program_state,
        #                                                                     self.evaluation_context,
        #                                                                     self.root_value_type)
        # action = SynthesisAction(directives, self.evaluation_context)
        # self.actions = [action]

    def get_actions(self) -> List[Action]:
        # TODO orchestrate subgoals
        actions_local = self.actions
        self.actions = []
        return actions_local


class PredicateSynthesisGoal(Goal):

    def __init__(self, parent: Goal,
                 draft_program: ProgramHypothesis,
                 evaluation_context: EvaluationContext):
        self.draft_program = draft_program
        self.evaluation_context = copy(evaluation_context)
        self.actions = []
        self.bootstrapped = False
        self.evaluation_context.predicate_mode = True
        super().__init__(parent)

    def accomplished(self, program_state: ProgramState) -> bool:

        previous_result_traces = [] if len(self._action_history) == 0 else [inner_trace for tup in self._action_history[-1][1] for inner_trace in tup[1]]
        is_complete = predicate_complete_condition(program_state, self.evaluation_context,
                                                   self.draft_program, previous_result_traces)
        if not is_complete and not self.bootstrapped:
            bootstrap_trace(program_state, self.evaluation_context, ArcObject, True)
            self.bootstrapped = True

        if not is_complete:
            self.__enqueue_actions(program_state)

        return is_complete

    def abandoned(self, program_state: ProgramState) -> bool:
        return False

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        pass

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        pass

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:
        return []

    def __enqueue_actions(self, program_state: ProgramState):
        directives: List[SynthesisDirective] = predicate_directive_selection(program_state,
                                                                             self.evaluation_context,
                                                                             self.draft_program)
        action = SynthesisAction(directives, self.evaluation_context)
        self.actions = [action]

    def get_actions(self) -> List[Action]:
        actions_local = self.actions
        self.actions = []
        return actions_local


class SynthesisGoal(Goal):

    def __init__(self, parent: Goal,
                 foundational_input_perception_model: PerceptionModel,
                 foundational_output_perception_model: PerceptionModel):
        self.foundational_input_perception_model = foundational_input_perception_model
        self.foundational_output_perception_model = foundational_output_perception_model

        self.evaluation_context = EvaluationContext(foundational_input_perception_model.id, foundational_output_perception_model.id)
        self.draft_program = ProgramHypothesis(foundational_input_perception_model, foundational_output_perception_model)

        self.operators_complete = False
        self.predicates_complete = False
        self.env_operators_complete = False

        self.any_subgoals_abandoned = False

        super().__init__(parent)

    def accomplished(self, program_state: ProgramState) -> bool:
        return self.env_operators_complete

    def abandoned(self, program_state: ProgramState) -> bool:
        return self.any_subgoals_abandoned

    def record_subgoal_success(self, program_state: ProgramState, subgoal: 'Goal'):
        if isinstance(subgoal, (OperatorSynthesisGoal, OperatorSynthesisExploreGoal, OperatorOrchestrateSynthesisGoal)):
            if subgoal.root_value_type == ArcObject:
                if isinstance(subgoal, (OperatorSynthesisExploreGoal, OperatorOrchestrateSynthesisGoal)):
                    self.operators_complete = operator_complete_condition(program_state, self.evaluation_context, self.draft_program,
                                                ArcObject)
                    print("OP COMPLETED FOR REAL", self.operators_complete)
                else:
                    self.operators_complete = True
            elif subgoal.root_value_type == EnvironmentShape:
                if isinstance(subgoal, (OperatorSynthesisExploreGoal, OperatorOrchestrateSynthesisGoal)):
                    self.env_operators_complete = operator_complete_condition(program_state, self.evaluation_context, self.draft_program,
                                                EnvironmentShape)
                else:
                    self.env_operators_complete = True
        elif isinstance(subgoal, PredicateSynthesisGoal):
            self.predicates_complete = True

    def record_subgoal_abandonment(self, subgoal: 'Goal'):
        self.any_subgoals_abandoned = True

    def get_new_subgoals(self, program_state: ProgramState) -> List[Goal]:
        subgoals = []

        if not self.operators_complete:
            subgoals.append(OperatorSynthesisGoal(
                parent=self,
                draft_program=self.draft_program,
                evaluation_context=self.evaluation_context,
                root_value_type=ArcObject
            ))
            # temp_root_trace = bootstrap_trace(program_state, self.evaluation_context, ArcObject, False)
            # subgoals.append(OperatorSynthesisExploreGoal(
            #     parent=self,
            #     evaluation_context=self.evaluation_context,
            #     root_value_type=ArcObject,
            #     root_trace=temp_root_trace,
            #     focus_node=temp_root_trace.graph.get_node_by_id(temp_root_trace.focus_node_id),
            #     perceived_subset=None,
            #     allowed_delta_depth=6
            # ))
            # subgoals.append(OperatorOrchestrateSynthesisGoal(
            #     parent=self,
            #     draft_program=self.draft_program,
            #     evaluation_context=self.evaluation_context,
            #     root_value_type=ArcObject
            # ))
        elif not self.predicates_complete:
            # NOTE passing previous goal action history's last element
            subgoals.append(PredicateSynthesisGoal(
                parent=self,
                draft_program=self.draft_program,
                evaluation_context=self.evaluation_context
            ))
        elif not self.env_operators_complete:
            subgoals.append(OperatorSynthesisGoal(
                parent=self,
                draft_program=self.draft_program,
                evaluation_context=self.evaluation_context,
                root_value_type=EnvironmentShape
            ))
            # temp_root_trace = bootstrap_trace(program_state, self.evaluation_context, EnvironmentShape, False)
            # subgoals.append(OperatorSynthesisExploreGoal(
            #     parent=self,
            #     evaluation_context=self.evaluation_context,
            #     root_value_type=EnvironmentShape,
            #     root_trace=temp_root_trace,
            #     focus_node=temp_root_trace.graph.get_node_by_id(temp_root_trace.focus_node_id),
            #     perceived_subset=None,
            #     allowed_delta_depth=2
            # ))
            # subgoals.append(OperatorOrchestrateSynthesisGoal(
            #     parent=self,
            #     draft_program=self.draft_program,
            #     evaluation_context=self.evaluation_context,
            #     root_value_type=EnvironmentShape
            # ))

        return subgoals

    def get_actions(self) -> List[Action]:
        return []


class GoalGraph:

    def __init__(self, root_goal: Goal):
        self.__root_goal: RootGoal = root_goal
        self.__goal_index: Dict[int, Goal] = {root_goal.id: root_goal}

        self.__active_goals: List[Goal] = [root_goal]
        self.__goals_to_active_subgoals: DefaultDict[int, List[Goal]] = defaultdict(list)

    def get_active_goals(self) -> List[Goal]:
        return self.__active_goals

    def refresh(self, program_state: ProgramState, last_action_goal: Goal, recursive = False):
        # Track processed goals to avoid infinite loops
        processed_goals = set()

        # Safety limit to prevent infinite loops - this is the max depth of goal chains
        max_iterations = 100

        for iteration in range(max_iterations):
            # Find newly completed goals that haven't been processed yet
            newly_completed = []
            for goal in list(self.__active_goals):  # Use list() to avoid modification during iteration
                if goal.id not in processed_goals and goal.accomplished(program_state):
                    print("ACTIONS TAKEN", goal.__class__, len(goal.get_action_history()))
                    newly_completed.append(goal)
                    processed_goals.add(goal.id)

            if not newly_completed:
                break  # No more completed goals to process

            # Process each completed goal
            for goal in newly_completed:
                self.__active_goals.remove(goal)
                parent_id = goal.get_parent_id()

                # Clean up this goal's entry in __goals_to_active_subgoals
                if goal.id in self.__goals_to_active_subgoals:
                    del self.__goals_to_active_subgoals[goal.id]

                if parent_id is not None:
                    sibling_goals = self.__goals_to_active_subgoals[parent_id]
                    if goal in sibling_goals:
                        sibling_goals.remove(goal)

                    parent_goal = self.__goal_index[parent_id]
                    parent_goal.record_subgoal_success(program_state, goal)

                    # If parent has no more active subgoals, reactivate it
                    if len(self.__goals_to_active_subgoals[parent_id]) == 0 and parent_goal not in self.__active_goals:
                        self.__active_goals.append(parent_goal)
                        grandparent_id = parent_goal.get_parent_id()
                        if grandparent_id is not None:
                            if parent_goal not in self.__goals_to_active_subgoals[grandparent_id]:
                                self.__goals_to_active_subgoals[grandparent_id].append(parent_goal)

        # Check for abandoned goals
        for iteration_abandoned in range(max_iterations):
            newly_abandoned = []
            for goal in list(self.__active_goals):
                if goal.id not in processed_goals and goal.abandoned(program_state):
                    print("GOAL ABANDONED", goal.id)
                    newly_abandoned.append(goal)
                    processed_goals.add(goal.id)

            if not newly_abandoned:
                break

            for goal in newly_abandoned:
                self.__active_goals.remove(goal)
                parent_id = goal.get_parent_id()

                if goal.id in self.__goals_to_active_subgoals:
                    del self.__goals_to_active_subgoals[goal.id]

                if parent_id is not None:
                    sibling_goals = self.__goals_to_active_subgoals[parent_id]
                    if goal in sibling_goals:
                        sibling_goals.remove(goal)

                    parent_goal = self.__goal_index[parent_id]
                    parent_goal.record_subgoal_abandonment(goal)

                    if len(self.__goals_to_active_subgoals[parent_id]) == 0 and parent_goal not in self.__active_goals:
                        self.__active_goals.append(parent_goal)
                        grandparent_id = parent_goal.get_parent_id()
                        if grandparent_id is not None:
                            if parent_goal not in self.__goals_to_active_subgoals[grandparent_id]:
                                self.__goals_to_active_subgoals[grandparent_id].append(parent_goal)

        # Now handle spawning new subgoals for active goals
        active_goals_copy: List[Goal] = list(self.__active_goals)
        for active_goal in active_goals_copy:
            if active_goal.id not in processed_goals:  # Don't spawn subgoals for goals we just completed
                next_subgoals = active_goal.get_new_subgoals(program_state)
                if len(next_subgoals) > 0:
                    for subgoal in next_subgoals:
                        self.__goal_index[subgoal.id] = subgoal
                        self.__goals_to_active_subgoals[active_goal.id].append(subgoal)
                        self.__active_goals.append(subgoal)
                    self.__active_goals.remove(active_goal)


