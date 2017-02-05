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
 * This function is intended for use with CyKeplerSTM 
 * See CyKeplerSTM.pyx/CyKeplerSTM_setup.py for compilation notes.
 *
 *=================================================================*/

#include <math.h>
#include <stdio.h>
#include <string.h>

#if !defined(EPS)
#define EPS(X) pow(2,log(fabs(X))/log(2) - 52.0)
#endif

double DOT(double x1[], double x2[], int size) {
    double res = 0;
    int i;
    for (i=0; i<size; i++) {
        res += x1[i]*x2[i];
    }
    return res;
}


int KeplerSTM_C (double x0[], double dt, double mu, double x1[] ){
    
    /* Initialize orbit values*/
    double r0[3] = {x0[0],x0[1],x0[2]};
    double v0[3] = {x0[3],x0[4],x0[5]};
    
    double r0norm =  sqrt(pow(r0[0], 2)+pow(r0[1], 2)+pow(r0[2], 2));
    double nu0 = DOT(r0,v0,3);
    double beta = 2.0*mu/r0norm - DOT(v0,v0,3);

    /* For elliptic orbits, account for period effects */
    double deltaU = 0;
    if (beta > 0){
        double P = 2.0*M_PI*mu*pow(beta,(-3.0/2.0));
        double norb = floor((dt + P/2.0 - 2.0*nu0/beta)/P);
        deltaU = 2.0*M_PI*norb*pow(beta,(-5.0/2.0));       
    }
    
    /* Initialize continued fraction values*/
    double a = 5.0;
    double b = 0.0;
    double c = 5.0/2.0;
    double k, l, d, n;
    
    /*kepler iteration loop
     *loop until convergence of the time array to the time step*/
    double t = 0;
    int counter = 0;
    int counter2 = 0;
    double tol = EPS(dt);
    double u = 0;
    double q, U0w2, U1w2, U, U0, U1, U2, U3, r, A, B, cf, cfprev;
    while ((fabs(t-dt) > tol) && (counter < 1000)){
        q = beta*pow(u,2.0)/(1+beta*pow(u,2.0));
        
        /* initialize continued fractions */
        A = 1.0;
        B = 1.0;
        cf = 1.0;
        cfprev = 2.0;
        counter2 = 0;
        k = 1.0 - 2.0*(a-b);
        l = 2.0*(c-1.0);
        d = 4.0*c*(c-1.0);
        n = 4.0*b*(c-a);

        /* loop until convergence of continued fraction*/
        while ((fabs(cf-cfprev) > EPS(cf)) && (counter2 < 1000)){
            k = -k;
            l += 2.0;
            d += 4.0*l;
            n += (1.0+k)*l;
            A = d/(d - n*A*q);
            B = (A-1.0)*B;
            cfprev = cf;
            cf += B;
            counter2 += 1;
        }
        if (counter2 == 1000){
            printf("Failed to converge on continued fraction.");
            return -1;
        }

        U0w2 = 1.0 - 2.0*q;
        U1w2 = 2.0*(1-q)*u;
        U = (16.0/15.0)*pow(U1w2,5.0)*cf + deltaU;     
        U0 = 2.0*pow(U0w2,2.0)-1.0;
        U1 = 2.0*U0w2*U1w2;
        U2 = 2.0*pow(U1w2,2.0);
        U3 = beta*U + U1*U2/3.0;
        r = r0norm*U0 + nu0*U1 + mu*U2;
        t = r0norm*U1 + nu0*U2 + mu*U3;
        u -= (t-dt)/(4.0*(1-q)*r);
        counter += 1;
    }
    if (counter == 1000){
        printf("Failed to converge on time step.");
        printf("t-dt = %6.6e\n", t-dt);
        return -2;
    }
 

    double f = 1.0 - mu/r0norm*U2;
    double g = r0norm*U1 + nu0*U2;
    double F = -mu*U1/r/r0norm;
    double G = 1.0 - mu/r*U2;
        
    int i;
    for (i=0; i<3; i++) {
        x1[i] = x0[i]*f + x0[i+3]*g;
    }
    for (i=3; i<6; i++) {
        x1[i] = x0[i-3]*F + x0[i]*G;
    }
    
    return 0;
}
