import sys

content = open('env/mec_offloaing_envs/offloading_task_graph.py', 'r').read()
idx = content.find('encode_point_sequence_with_ranking_and_cost')
if idx != -1:
    print(f'Found at position: {idx}')
    # Find line number
    lines = content[:idx].count('\n')
    print(f'Line number: {lines + 1}')
else:
    print('Not found')