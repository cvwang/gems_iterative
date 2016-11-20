all:
	spark-submit --master local[4] GemsApp.py

diff_boundary:
	diff boundary_list.txt ../../matlab/boundary_list.txt

diff_output:
	diff output.txt ../../matlab/output.txt

diff_x:
	diff x.txt ../../matlab/x.txt