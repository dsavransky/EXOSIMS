/*=================================================================
 *
 * KeplerSTM_C.C	Apply KEPLER STM to propagate two body orbit
 *
 * KeplerSTM_C(X0,DT,MU,X1)
 *      X0      6  x 1 : [r_0, v_0]
 *      DT      float  : time step (t1 - t0)
 *      MU      float  : Gravitational paramter
 *      X1      6 x 1  : [r_1, v_1]
 *
 * Written by Dmitry Savransky (ds264@cornell.edu)
 * Algorithm from Shepperd, 1984 which employs Goodyear's universal 
 * variables and solves the Kepler problem using continued fractions.
 *
 *=================================================================*/

int KeplerSTM_C (double x0[], double dt, double mu, double x1[] );
