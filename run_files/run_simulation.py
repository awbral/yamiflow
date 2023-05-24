import json
from os.path import join, dirname
from analysis import Analysis
import sys

# import parameters
if len(sys.argv) == 2:
    parameter_file_name = join(dirname(__file__), sys.argv[1])
else:
    parameter_file_name = join(dirname(__file__), 'parameters.json')
with open(parameter_file_name, 'r') as parameter_file:
    parameters = json.load(parameter_file)

simulation = Analysis(parameters)
simulation.run()
# simulation.post_process()
