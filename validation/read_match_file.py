# %%
import sys
args = sys.argv[1:]

match_file = args[0]
match_key_points = 0

res = open(match_file,'r')

for x in res.readlines():
    match_key_points+= int(x.split(':')[-1])

print("Total Matching Pairs:",match_key_points)


