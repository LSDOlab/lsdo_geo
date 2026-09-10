"""
Script to regenerate the 5x10 LHS grid from saved screenshots without rerunning the full simulation.
Useful for tweaking visualization parameters without waiting for all 50 simulations to complete.
"""

from PIL import Image, ImageDraw
from pathlib import Path

# Parameters
n_fuselage_samples = 5
n_wing_samples = 10
screenshot_folder = Path("examples/additional_examples/hand_launched_uavs/lhs_sample_screenshots")
line_width = 3
line_color = 'black'

# Load first screenshot to get dimensions
first_screenshot = Image.open(screenshot_folder / "combinations" / "fuselage_00_wing_00.png")
screenshot_width, screenshot_height = first_screenshot.size

# Create blank canvas for full grid (with headers)
total_width = screenshot_width + n_wing_samples * screenshot_width
total_height = screenshot_height + n_fuselage_samples * screenshot_height
grid_image = Image.new('RGB', (total_width, total_height), color='white')
draw = ImageDraw.Draw(grid_image)

# Add wing header screenshots (column labels)
wing_header_folder = screenshot_folder / "wing_headers"
for col in range(n_wing_samples):
    wing_header_path = wing_header_folder / f"wing_{col:02d}.png"
    wing_header = Image.open(wing_header_path)
    wing_header = wing_header.resize((screenshot_width, screenshot_height))
    x = screenshot_width + col * screenshot_width
    y = 0
    grid_image.paste(wing_header, (x, y))

# Add fuselage header screenshots (row labels) and combination screenshots
fuselage_header_folder = screenshot_folder / "fuselage_headers"
for row in range(n_fuselage_samples):
    fuselage_header_path = fuselage_header_folder / f"fuselage_{row:02d}.png"
    fuselage_header = Image.open(fuselage_header_path)
    fuselage_header = fuselage_header.resize((screenshot_width, screenshot_height))
    x = 0
    y = screenshot_height + row * screenshot_height
    grid_image.paste(fuselage_header, (x, y))

# Load and paste combination screenshots
for row in range(n_fuselage_samples):
    for col in range(n_wing_samples):
        # Load screenshot
        screenshot_path = screenshot_folder / "combinations" / f"fuselage_{row:02d}_wing_{col:02d}.png"
        screenshot = Image.open(screenshot_path)
        
        # Paste into grid
        x = screenshot_width + col * screenshot_width
        y = screenshot_height + row * screenshot_height
        grid_image.paste(screenshot, (x, y))

# Draw dividing lines
# Horizontal line separating headers from combinations
y_line = screenshot_height
draw.rectangle(
    [(0, y_line), (total_width, y_line + line_width)],
    fill=line_color
)

# Vertical line separating fuselage headers from combinations
x_line = screenshot_width
draw.rectangle(
    [(x_line, 0), (x_line + line_width, total_height)],
    fill=line_color
)

# Save grid
grid_output_path = screenshot_folder / "hand_launched_uavs_separate_lhs_grid.png"
grid_image.save(str(grid_output_path))

print(f"Grid regenerated and saved to {grid_output_path}")
