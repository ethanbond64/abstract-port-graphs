## Abstract Port Graphs

Abstract Port Graphs (APG) framework for program synthesis that separates perception from symbolic reasoning. 
This work was developed as an approach to [ARC AGI](https://arcprize.org/).

### Paper
https://drive.google.com/file/d/1T1EuGF-lzQHQYAN18er3j-4KBi1gqtp-/view

## Contents
- APG framework (project root, classes outlined below)
- `arc/` - ARC Domain Specific Language.
- `arc/arc_solutions.py` - Test that runs 48 handwritten solutions to ARC tasks using the DSL.
- `debugger/debugger.py` - Visual debugger that runs as a local web app to view graphs.
- `synthesis/` - Program synthesis engine.
- `synthesis/arc_synthesis_tests.py` - Test that runs 11 ARC tasks which can be synthesized in under 1 second.


## Core Classes

- **Node** (`nodes.py`): Base class for all graph nodes - computational units with input ports and unique IDs.
- **PortGraph** (`port_graphs.py`): Symbolic programs represented with APG grammar.
- **PerceptionModel** (`programs.py`): Abstract class that converts raw data into perceived objects which are fed into the port graphs.
- **Program** (`programs.py`): Combines a perception model with a port graph to create an executable program.
- **Interpreter** (`interpreter/interpreter.py`): Executes programs by applying the PerceptionModel to raw data, and then feeding the perceived data structures through the port graphs in a dataflow fashion.
- **Library** (`base_library.py`): Registry for available node types and value types in the framework

## Example Usage

Complete example demonstrating all core classes:

```python
from typing import List, Any
from port_graphs import PortGraph
from programs import Program, PerceptionModel
from interpreter.interpreter import Interpreter
from nodes import InputNode, OutputNode, Constant
from operator_primitives import AdditionOperator, MultiplyOperator
from base_types import PerceivedType


# Define a custom perceived type
class DslNumber(PerceivedType):
    def __init__(self, value):
        self.value = value

    def get_perception_id(self):
        return "number"

# Define a custom perception model to convert raw data into perceived types
class SimplePerceptionModel(PerceptionModel):
    def apply_perception(self, raw_data: Any) -> List[PerceivedType]:
        return [DslNumber(x) for x in raw_data]
    

# Build a simple program that does the following calculation: input + 10
graph = PortGraph()

input_node = InputNode(int, perception_qualifiers={"number"})
constant_node = Constant(int, 10)
addition_node = AdditionOperator()
output_node = OutputNode(int)

graph.add_edge(input_node, addition_node, to_port=0)
graph.add_edge(constant_node, addition_node, to_port=1)

graph.add_edge(addition_node, output_node)

perception_model = SimplePerceptionModel()
program = Program(perception_model, graph)

# Execute with the interpreter
interpreter = Interpreter()
raw_data = [5, 7, 3]
result = interpreter.evaluate_program(program, raw_data)

# Result: {<output_node_id>: [15, 17, 13]}
```

## APG Grammar

### Source Nodes (Input)
| Node Type      | Description                       | Output Type    |
|----------------|-----------------------------------|----------------|
| `InputNode`    | Receives single perceived object  | Specified type |
| `InputSetNode` | Receives set of perceived objects | DslSet         |
| `Constant`     | Holds static value                | Specified type |

### Intermediate Nodes (Processing)
| Node Type                    | Description                                                                   | Input Ports | Output Type  |
|------------------------------|-------------------------------------------------------------------------------|-------------|--------------|
| **Operators**                |                                                                               |             |              |
| `AdditionOperator`           | Adds two integers                                                             | 2           | int          |
| `SubtractionOperator`        | Subtracts two integers                                                        | 2           | int          |
| `MultiplyOperator`           | Multiplies two numbers                                                        | 2           | Number       |
| `ConstructorOperator`        | Constructs custom type                                                        | Variable    | Custom type  |
| **Set Operators**            |                                                                               |             |              |
| `SumSetOperator`             | Sums integer set                                                              | 1           | int          |
| `SetRankOperator`            | Ranks element in set                                                          | 2           | int          |
| `ApplyScalarOpToSetOperator` | Maps operator over set                                                        | Variable    | DslSet       |
| `CreateLocalSetOperator`     | Creates singleton set                                                         | 1           | DslSet       |
| `AddToLocalSetOperator`      | Adds element to set                                                           | 2           | DslSet       |
| **Relationships**            |                                                                               |             |              |
| `Equals`                     | Tests equality                                                                | 2           | bool         |
| `NotEquals`                  | Tests inequality                                                              | 2           | bool         |
| `LessThan`                   | Tests less than                                                               | 2           | bool         |
| `LessThanOrEqual`            | Tests less than or equal                                                      | 2           | bool         |
| `GreaterThan`                | Tests greater than                                                            | 2           | bool         |
| `GreaterThanOrEqual`         | Tests greater than or equal                                                   | 2           | bool         |
| `SetContains`                | Tests set membership                                                          | 2           | bool         |
| `SetNotContains`             | Tests set non-membership                                                      | 2           | bool         |
| **Control Flow**             |                                                                               |             |              |
| `RecursiveProxyNode`         | Enables recursion                                                             | 2           | Any          |
| `IterativeProxyNode`         | Enables iteration                                                             | 2           | Any          |
| `SetJoin`                    | Aggregates values to set                                                      | 1           | DslSet       |
| `SetSplit`                   | Distributes set to instances                                                  | 1           | Element type |
| `DisjointSetNode`            | Share set across disjoint graphs  <br/> while still treating them as disjoint | 1           | DslSet       |

### Output Nodes
| Node Type | Description | Input Ports |
|-----------|-------------|-------------|
| `OutputNode` | Produces final result | 1 |


## Setup
- Use python 3.13+
- Install dependencies via `pip install -r requirements.txt`