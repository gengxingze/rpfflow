from itertools import combinations
from copy import deepcopy

indices = [0,1,2,3]
for idx1, idx2 in combinations(indices, 2):
    print(idx1, idx2)

print(list(combinations(indices, 2)))
print(list(combinations(indices, 3)))
