## Abstract Port Graphs

Abstract Port Graphs (APG) framework for program synthesis that separates perception from symbolic reasoning. 
This work was developed as an approach to [ARC AGI](https://arcprize.org/).

### Paper
TODO

## Contents
- APG framework (project root, classes outlined below)
- `arc/` - ARC Domain Specific Language.
- `arc/arc_solutions.py` - Test that runs 48 handwritten solutions to ARC tasks using the DSL.
- `debugger/debugger.py` - Visual debugger that runs as a local web app to view graphs.
- `synthesis/` - Program synthesis engine.
- `synthesis/arc_synthesis_tests.py` - Test that runs 11 ARC tasks which can be synthesized in under 1 second.


## Core Classes

### Node
Defined in `nodes.py` \
TODO describe


### PortGraph
Defined in `port_graphs.py` \
TODO describe
TODO example usage

### PerceptionModel
Defined in `programs.py`\
TODO just put the signature

### Program
Defined in `programs.py`\
TODO describe

### Interpreter
Defined in `interpreter/interpreter.py` \
TODO describe 
TODO example usage

## APG Grammar
TODO table of primitive nodes + types