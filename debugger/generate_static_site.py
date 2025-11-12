#!/usr/bin/env python3
"""
Static site generator for ARC Visualizer
Generates static HTML files that can be hosted without a Flask server
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add parent directory to path to import arc modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'debugger'))

# Import after path setup
from arc.arc_solutions import *
from arc.arc_utils import ArcTask
from arc.arc_objects import ArcGraph
from debugger_visualization import generate_graph_html
from arc_task_visualization import generate_task_html


def get_all_task_ids():
    """Get all task IDs from the ARC dataset."""
    json_path = os.path.join(os.path.dirname(__file__), '../arc/data/arc-agi_training_challenges.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    return sorted(data.keys())


def generate_graph_file(task_id, output_dir):
    """Generate the graph HTML file for a specific task."""
    try:
        function_name = "solve_" + task_id
        graph_function = globals().get(function_name)

        if graph_function is None:
            print(f"  Warning: No solution found for {task_id}")
            return False

        graph = graph_function()
        html_path = generate_graph_html(graph, None)

        # Read the generated HTML
        with open(html_path, 'r') as f:
            html_content = f.read()

        # Write to output directory
        output_path = os.path.join(output_dir, f"{task_id}_graph.html")
        with open(output_path, 'w') as f:
            f.write(html_content)

        # Clean up temporary file
        os.remove(html_path)

        return True
    except Exception as e:
        print(f"  Error generating graph for {task_id}: {e}")
        return False


def generate_grid_file(task_id, output_dir):
    """Generate the grid visualization HTML file for a specific task."""
    try:
        html_content = generate_task_html(task_id)

        # Wrap the content in a complete HTML document
        full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARC Task {task_id}</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>'''

        output_path = os.path.join(output_dir, f"{task_id}_grids.html")
        with open(output_path, 'w') as f:
            f.write(full_html)

        return True
    except Exception as e:
        print(f"  Error generating grids for {task_id}: {e}")
        return False


def generate_static_index(task_ids, output_dir):
    """Generate the static index.html file."""

    # Generate options for the dropdown
    options_html = ['<option value="">Select...</option>']
    for task_id in task_ids:
        options_html.append(f'<option value="{task_id}">{task_id}</option>')
    options_str = '\n                                '.join(options_html)

    index_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ARC Visualizer - Graph Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }}

        .main-container {{
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            gap: 20px;
        }}

        .left-content {{
            flex: 1;
            min-width: 0;
        }}

        .right-sidebar {{
            width: 35%;
            min-width: 350px;
        }}

        .header {{
            background: white;
            border-radius: 8px;
            padding: 15px 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            margin-bottom: 15px;
        }}

        .header-content {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }}

        .title-section {{
            flex-shrink: 0;
        }}

        h1 {{
            color: #333;
            font-size: 24px;
            margin-bottom: 2px;
        }}

        .subtitle {{
            color: #666;
            font-size: 12px;
        }}

        .controls-section {{
            display: flex;
            gap: 15px;
            align-items: center;
            flex: 1;
        }}

        .form-inline {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        label {{
            font-weight: 600;
            color: #333;
            font-size: 14px;
            white-space: nowrap;
        }}

        select {{
            width: 200px;
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            background: white;
            transition: all 0.2s ease;
            cursor: pointer;
        }}

        select:hover {{
            border-color: #4a90e2;
        }}

        select:focus {{
            outline: none;
            border-color: #4a90e2;
            box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
        }}

        button {{
            padding: 6px 18px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .btn-primary {{
            background: #4a90e2;
            color: white;
        }}

        .btn-primary:hover {{
            background: #357abd;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
        }}

        .graph-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        }}

        .graph-header {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #f0f0f0;
        }}

        iframe {{
            width: 100%;
            height: 600px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
        }}

        .data-output {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            height: calc(100vh - 40px);
            position: sticky;
            top: 20px;
            display: flex;
            flex-direction: column;
        }}

        .data-header {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid #f0f0f0;
            flex-shrink: 0;
        }}

        .task-iframe {{
            width: 100%;
            height: calc(100% - 45px);
            border: none;
            background: white;
            border-radius: 6px;
        }}

        .empty-state {{
            color: #999;
            font-style: italic;
            text-align: center;
            padding: 20px;
        }}

        /* Responsive adjustments */
        @media (max-width: 1200px) {{
            .main-container {{
                flex-direction: column;
            }}

            .right-sidebar {{
                width: 100%;
            }}

            .data-output {{
                height: auto;
                max-height: 400px;
                position: static;
            }}
        }}

        @media (max-width: 768px) {{
            .header-content {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .controls-section {{
                width: 100%;
                flex-direction: column;
                align-items: stretch;
            }}

            .form-inline {{
                flex-direction: column;
                align-items: stretch;
            }}

            select {{
                width: 100%;
            }}
        }}
    </style>
    <script>
        function loadChallenge() {{
            const select = document.getElementById('challenge_id');
            const challengeId = select.value;

            const graphFrame = document.getElementById('graph-frame');
            const taskFrame = document.getElementById('task-frame');
            const taskContainer = document.getElementById('task-container');
            const emptyState = document.getElementById('empty-state');

            if (challengeId) {{
                // Update graph iframe
                graphFrame.src = challengeId + '_graph.html';

                // Update task iframe
                taskFrame.src = challengeId + '_grids.html';
                taskContainer.style.display = 'block';
                emptyState.style.display = 'none';
            }} else {{
                // Clear iframes
                graphFrame.src = 'about:blank';
                taskFrame.src = 'about:blank';
                taskContainer.style.display = 'none';
                emptyState.style.display = 'block';
            }}
        }}
    </script>
</head>
<body>
    <div class="main-container">
        <div class="left-content">
            <div class="header">
                <div class="header-content">
                    <div class="title-section">
                        <h1>ARC Visualizer</h1>
                        <p class="subtitle">Graph structure visualization</p>
                    </div>

                    <div class="controls-section">
                        <div class="form-inline">
                            <label for="challenge_id">Challenge:</label>
                            <select id="challenge_id" onchange="loadChallenge()">
                                {options_str}
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            <div class="graph-container">
                <div class="graph-header">Graph Visualization</div>
                <iframe id="graph-frame" src="about:blank" title="Graph Viewer"></iframe>
            </div>
        </div>

        <div class="right-sidebar">
            <div class="data-output">
                <div class="data-header">ARC Task Display</div>
                <div id="task-container" style="display: none; height: calc(100% - 45px);">
                    <iframe id="task-frame" src="about:blank" class="task-iframe" title="ARC Task Visualization"></iframe>
                </div>
                <div id="empty-state" class="empty-state">Select a challenge to view the ARC task</div>
            </div>
        </div>
    </div>
</body>
</html>'''

    output_path = os.path.join(output_dir, 'index.html')
    with open(output_path, 'w') as f:
        f.write(index_html)

    print("Generated index.html")


def main():
    """Main function to generate all static files."""

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Get all task IDs
    all_task_ids = get_all_task_ids()
    print(f"Found {len(all_task_ids)} tasks")
    print()

    # Filter to only tasks that have solutions
    available_task_ids = []
    for task_id in all_task_ids:
        function_name = "solve_" + task_id
        if globals().get(function_name) is not None:
            available_task_ids.append(task_id)

    print(f"Found solutions for {len(available_task_ids)} tasks")
    print()

    # Generate files for each task
    graph_success = 0
    grid_success = 0

    for i, task_id in enumerate(available_task_ids, 1):
        print(f"Processing {task_id} ({i}/{len(available_task_ids)})...")

        if generate_graph_file(task_id, output_dir):
            graph_success += 1

        if generate_grid_file(task_id, output_dir):
            grid_success += 1

    print()
    print(f"Generated {graph_success} graph files")
    print(f"Generated {grid_success} grid files")

    # Generate index.html
    generate_static_index(available_task_ids, output_dir)

    print()
    print(f"Static site generation complete!")
    print(f"Files saved to: {output_dir}")
    print(f"You can now upload the contents of the 'out' directory to S3 or any static hosting service.")


if __name__ == "__main__":
    main()