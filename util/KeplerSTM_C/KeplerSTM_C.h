/*=================================================================
 *
 * KeplerSTM_C.C	Apply KEPLER STM to propagate two body orbit
 *
 * KeplerSTM_C(X0,DT,MU,X1)
 *      X0      6  x 1 : [r_0, v_0]
 *      DT      float  : time step (t1 - t0)
 *      MU      float  : Gravitational paramter
 *      X1      6 x 1  : [r_1, v_1]
 *      EPSMULT float  : tolerance parameter
 *
 * Written by Dmitry Savransky (ds264@cornell.edu)
 * Two algorithms are implemented, both using Batting/Goodyear universal variables. 
 * The first is from Shepperd (1984), using continued fraction to solve the Kepler equation.
 * The second is from Vallado (2004), using Newton iteration to solve the time equation. 
 * One algorithm is used preferentially, and the other is called only in the case of convergence
 * failure on the first.  All convergence is calculated to machine precision of the data type and 
 * variable size, scaled by a user-selected multiple.
 *=================================================================*/

int KeplerSTM_C (double x0[], double dt, double mu, double x1[], double epsmult);

int KeplerSTM_C_vallado (double x0[], double dt, double mu, double x1[], double epsmult);
