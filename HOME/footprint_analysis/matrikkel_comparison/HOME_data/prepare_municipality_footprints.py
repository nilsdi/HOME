"""
Providing the data (both for maps and a line plot) for the development of built-up area
within the boundaries of a municipality.
"""

# %%
import matplotlib.pyplot as plt

from pathlib import Path

# %%
root_dir = Path(__file__).parents[4]
print(root_dir)

# %%

# Pseudo code of what we need and do.

# final result: need the total area of footprints over time, where all locations are updated 
# with the most recent project.
# also need this information with high spatial resolution.

# without even reading the footprints, we just overlap the municipality boundaries with 
# the project coverage to establish what is covered. i
# if we sort the projects by date, we can take the first one, and make a new boundary for the 
# earliest coverage (and how long this boundary is valid). we can then overlap with 
# the next project, reducing the size of the earlier boundaries. In the end, we have a large set
# of boundaries tied to a project, and a time period.
# after that we can go through the projects, and read out the information (gridded or non gridded)
# for each "boundary" of the project. 
# We then go through time and establish the total number of periods with stable coverage.
# We can then go through these periods and sum the entries of the boundaries and hove something we can plot.

# for plotting: we can determine some uncertainty by checking the difference for each 
# covered area (grid cells?) between two updates and treat those as upper and lower bounds.