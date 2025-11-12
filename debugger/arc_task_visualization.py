import json
import os
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# ARC color scheme (standard colors for digits 0-9)
ARC_COLORS = [
    '#000000',  # 0 - Black
    '#0074D9',  # 1 - Blue
    '#FF4136',  # 2 - Red
    '#2ECC40',  # 3 - Green
    '#FFDC00',  # 4 - Yellow
    '#AAAAAA',  # 5 - Gray
    '#F012BE',  # 6 - Magenta
    '#FF851B',  # 7 - Orange
    '#7FDBFF',  # 8 - Light Blue
    '#870C25',  # 9 - Maroon
]

def load_arc_task(task_id):
    """Load a specific ARC task from the JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), '../arc/data/arc-agi_training_challenges.json')

    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data.get(task_id)


def draw_grid(ax, grid, title):
    """Draw a single grid with ARC colors."""
    grid = np.array(grid)
    height, width = grid.shape

    # Create a color map from ARC colors
    cmap = ListedColormap(ARC_COLORS)

    # Draw the grid
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation='nearest')

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='#222222', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)

    # Remove tick labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Add title
    ax.set_title(title, fontsize=9, fontweight='bold', pad=3)

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')


def generate_task_visualization(task_id):
    """Generate a matplotlib figure showing all train and test examples for a task."""
    task = load_arc_task(task_id)

    if not task:
        return None

    # Count total examples
    train_examples = task.get('train', [])
    test_examples = task.get('test', [])
    total_examples = len(train_examples) + len(test_examples)

    if total_examples == 0:
        return None

    # Create figure with subplots
    # Each row shows one example (input on left, output on right)
    fig = plt.figure(figsize=(8, 1.5 * total_examples))
    fig.suptitle(f'ARC Task: {task_id}', fontsize=11, fontweight='bold', y=1.01)

    subplot_idx = 1

    # Draw training examples
    for i, example in enumerate(train_examples):
        # Input grid
        ax_input = fig.add_subplot(total_examples, 2, subplot_idx)
        draw_grid(ax_input, example['input'], f'Train {i+1} - Input')
        subplot_idx += 1

        # Output grid
        ax_output = fig.add_subplot(total_examples, 2, subplot_idx)
        draw_grid(ax_output, example['output'], f'Train {i+1} - Output')
        subplot_idx += 1

    # Draw test examples
    for i, example in enumerate(test_examples):
        # Input grid
        ax_input = fig.add_subplot(total_examples, 2, subplot_idx)
        draw_grid(ax_input, example['input'], f'Test {i+1} - Input')
        subplot_idx += 1

        # Output grid (may not exist for test)
        ax_output = fig.add_subplot(total_examples, 2, subplot_idx)
        if 'output' in example and example['output']:
            draw_grid(ax_output, example['output'], f'Test {i+1} - Output')
        else:
            # Draw a small placeholder grid with question mark
            placeholder_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 3x3 empty grid
            draw_grid(ax_output, placeholder_grid, f'Test {i+1} - Output')
            ax_output.text(0.5, 0.5, '?', fontsize=20, ha='center', va='center',
                          transform=ax_output.transAxes, color='gray', weight='bold')
        subplot_idx += 1

    plt.tight_layout(h_pad=0.5, w_pad=0.5)

    # Convert to base64 for embedding in HTML
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return image_base64


def generate_task_html(task_id):
    """Generate HTML with embedded task visualization."""
    image_base64 = generate_task_visualization(task_id)

    if not image_base64:
        return f'<div style="padding: 20px; text-align: center; color: #666;">Task {task_id} not found</div>'

    html = f'''
    <div style="width: 100%; height: 100%; overflow-y: auto; padding: 10px; background: white;">
        <img src="data:image/png;base64,{image_base64}" style="width: 100%; height: auto;" />
    </div>
    '''

    return html