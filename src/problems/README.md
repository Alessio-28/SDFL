## TEST PROBLEMS collection
The file <tt>probs.py</tt> exports the following objects:
- <tt>problem</tt> : a python class
- <tt>prob_collection</tt> : A dictionary of entries "probname" => problem object

A problem object is a structured type that has the following attributes:
- name   : string - name of the problem
- startp : numpy array - the starting point for the continuous problem
- n      : int - the total number of variables (>= 4)
- feval  : function handle - function to compute the objective function value
