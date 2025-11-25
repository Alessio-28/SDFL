## TEST PROBLEMS collection
The file <tt>probs.py</tt> exports the following objects:
- <tt>problem</tt> : a python class
- <tt>prob_collection</tt> : A dictionary of entries "probname" => problem object

A problem object is a structured type that has the following attributes:
- name   : string - name of the problem
- startp : numpy array - the starting point for the continuous problem
- lb     : numpy array - the lower bounds of the continuous problem
- ub     : numpy array - the upper bounds of the continuous problem
- n      : int - the total number of variables (>= 4)
- nint   : int - the number of discrete variables (>= 2)
- ncont  : int - the number of continuous variables (>= 2), N.B. n = nint+ncont<br>
```     
         BEWARE the variables are so intended
         x[0] ... x[ncont-1]     are the continuous variables
         x[ncont] ... x[n-1]     are the discrete variables
```
- lbmix  : numpy array - the actual lower bounds of the mixed integer problem
- ubmix  : numpy array - the actual upper bounds of the mixed integer problem
- x_initial : numpy array - the actual initial point of the mixed integer problem
- xmix   : numpy array - a temporary array used for computation
- feval  : function handle - function to compute the objective function value
```
          N.B. the point must be reconstructed through the use of
               reconstruct_xmix before calling feval!
```
- m      : dictionary with entries char => number of constraints
```
          it is equal to {'a': n-2, 'b': n-2, 'c': n-1, 'd': n-1,
                          'e': n-2, 'f': 1, 'z': 0}
          it is used to record the number of constraints for the given problem
          and for each of the six families of constraints a,b,c,d,e,f.
          'z' means no constraints
```
