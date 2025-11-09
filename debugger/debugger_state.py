import threading
from enum import Enum
from typing import List, Tuple, Optional, Any, Callable

import numpy as np

from arc.arc_utils import ArcTask
from interpreter.interpretter_common import GraphInstanceState
from arc.arc_objects import ArcGraph
from debugger_visualization import generate_graph_html

CLIENT_SIDE_TIMEOUT = 5


class DebuggerStage(Enum):
    RUN_REQUESTED = 0
    RUNNING = 1
    SUSPENDED = 2


class DebuggerState:
    # STATIC VARIABLES SET ON STARTUP
    __debug_mode: bool = False
    __challenge_id: str = None
    __case_id: int = 0
    __graph: ArcGraph = None

    # HTML GRAPH LOCATION
    __html_graph_location: str = None

    # THREAD MANAGEMENT
    # __stage: DebuggerStage = DebuggerStage.RUN_REQUESTED
    __running: bool = False
    __condition = threading.Condition()
    __interpreter_thread: threading.Thread = None

    # VARIABLES SET FROM FRONTEND
    __desired_node_id: int = None
    __desired_subgraph_instance_index: int = 0

    __temp_seen_nodes = set()

    # VARIABLES SET FROM BACKEND
    __current_node_id: int = None
    __current_subgraph_instance_index: int = 0
    __all_subgraph_instances: List[Tuple[dict, Optional[Any]]] = []  # List of tuples (subgraph instance, result)

    @staticmethod
    def initialize(challenge: ArcTask, case_id: int, graph: ArcGraph, solve_function: Callable):

        # If there is an existing thread, let it run to completion
        if DebuggerState.__interpreter_thread is not None and DebuggerState.__interpreter_thread.is_alive():
            DebuggerState.__debug_mode = False
            with DebuggerState.__condition:
                DebuggerState.__condition.notify_all()
            DebuggerState.__interpreter_thread.join()

        # Effectively final variables
        DebuggerState.__debug_mode = True
        DebuggerState.__challenge_id = challenge.challenge_id
        DebuggerState.__case_id = case_id
        DebuggerState.__graph = graph

        # Stop the interpreter thread on the first node
        DebuggerState.__desired_node_id = min(DebuggerState.__graph.graph.get_nodes_by_id().keys())
        DebuggerState.__desired_subgraph_instance_index = 0

        if DebuggerState.__case_id < len(challenge.train):
            input_matrix = challenge.train[DebuggerState.__case_id].input_matrix
        else:
            input_matrix = challenge.test[DebuggerState.__case_id - len(challenge.train)].input_matrix

        # Start the thread
        DebuggerState.__running = True
        DebuggerState.__interpreter_thread = threading.Thread(target=solve_function,
                                                              args=(DebuggerState.__graph, np.array(input_matrix) + 1))
        DebuggerState.__interpreter_thread.start()

        # If the thread is still running wait. The first thing the interpreter will do is
        # stop running, notify, and wait.
        with DebuggerState.__condition:
            while DebuggerState.__client_wait_condition():
                DebuggerState.__condition.wait(timeout=CLIENT_SIDE_TIMEOUT)

    @staticmethod
    def get_info():
        return {
            "challenge_id": DebuggerState.__challenge_id,
            "case_id": DebuggerState.__case_id,
            "running": DebuggerState.__running,
            "alive": DebuggerState.__interpreter_thread and DebuggerState.__interpreter_thread.is_alive(),
            "desired_node_id": DebuggerState.__desired_node_id,
            "desired_subgraph_instance_index": DebuggerState.__desired_subgraph_instance_index,
            "current_node_id": DebuggerState.__current_node_id,
            "current_subgraph_instance_index": DebuggerState.__current_subgraph_instance_index,
            "all_subgraph_instances": DebuggerState.__all_subgraph_instances
        }

    @staticmethod
    def get_graph_url():

        if DebuggerState.__html_graph_location is None:
            DebuggerState.__html_graph_location = generate_graph_html(DebuggerState.__graph,
                                                                      DebuggerState.__current_node_id)

        return DebuggerState.__html_graph_location

    @staticmethod
    def set_debugger_commands(desired_node_id: int, desired_subgraph_instance_index: int):
        print("enter debugger commands")
        with DebuggerState.__condition:
            while DebuggerState.__client_wait_condition():
                DebuggerState.__condition.wait(timeout=CLIENT_SIDE_TIMEOUT)
            # print("Thread alive:", DebuggerState.__interpreter_thread.is_alive())

            DebuggerState.__desired_node_id = desired_node_id
            DebuggerState.__desired_subgraph_instance_index = desired_subgraph_instance_index
            # DebuggerState.__stage = DebuggerStage.RUN_REQUESTED

            # Notify interpreter thread to start running with new instructions.
            # Wait until interpreter notifies back to exit this method.
            DebuggerState.__condition.notify_all()

            while DebuggerState.__client_wait_condition():
                DebuggerState.__condition.wait(timeout=CLIENT_SIDE_TIMEOUT)

    @staticmethod
    def starter_breakpoint():
        if DebuggerState.__debug_mode:
            DebuggerState.__breakpoint()

    @staticmethod
    def try_breakpoint(current_node_id: int, current_subgraph_instance_index: int,
                       subgraph_instances: List[GraphInstanceState], results: List[Any]):
        # print("enter breakpoint")
        if DebuggerState.__debug_mode:

            if current_node_id not in DebuggerState.__temp_seen_nodes:
                DebuggerState.__temp_seen_nodes.add(current_node_id)

                DebuggerState.__current_node_id = current_node_id
                DebuggerState.__current_subgraph_instance_index = current_subgraph_instance_index

                DebuggerState.__html_graph_location = generate_graph_html(DebuggerState.__graph,
                                                                          DebuggerState.__current_node_id)

                DebuggerState.__breakpoint()

    @staticmethod
    def __breakpoint():
        if DebuggerState.__debug_mode:
            with DebuggerState.__condition:
                DebuggerState.__running = False
                DebuggerState.__condition.notify_all()
                DebuggerState.__condition.wait()
                DebuggerState.__running = True

    @staticmethod
    def __client_wait_condition():
        return (DebuggerState.__running and DebuggerState.__interpreter_thread is not None and
                DebuggerState.__interpreter_thread.is_alive())
