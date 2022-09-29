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

#if !defined(MAX)
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

#if !defined(M_PI)
#define M_PI        3.14159265358979323846264338327950288   /* pi             */
#endif

double DOT(double x1[], double x2[], int size) {
    double res = 0;
    int i;
    for (i=0; i<size; i++) {
        res += x1[i]*x2[i];
    }
    return res;
}


int KeplerSTM_C (double x0[], double dt, double mu, double x1[], double epsmult){
    
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
    while ((fabs(t-dt) > epsmult*tol) && (counter < 1000)){
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
        while ((fabs(cf-cfprev) > epsmult*EPS(cf)) && (counter2 < 1000)){
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
            /*printf("Failed to converge on continued fraction.");*/
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
        /*printf("Failed to converge on time step.");
        printf("t-dt = %6.6e\n", t-dt);*/
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

int KeplerSTM_C_vallado (double x0[], double dt, double mu, double x1[], double epsmult){
    
    double epsval = 1.0e-12;

    /* Initialize orbit values*/
    double r0[3] = {x0[0],x0[1],x0[2]};
    double v0[3] = {x0[3],x0[4],x0[5]};
    
    double r0norm =  sqrt(pow(r0[0], 2)+pow(r0[1], 2)+pow(r0[2], 2));
    double v0norm2 =  DOT(v0,v0,3);
    double nu0 = DOT(r0,v0,3);
    double beta = 2.0*mu/r0norm - v0norm2;
    double alpha = beta/mu;
    double nu0osmu = nu0/sqrt(mu);

    /*initialize universal var*/
    double xi;
    if (alpha >= epsval){
        /* ellipses */
        xi = sqrt(mu*dt*alpha);
        if (fabs(alpha - 1.0) > epsval){
            /* near circs */
            xi *= 0.97;
        }
    } else if (fabs(alpha) < epsval){
        /* parabolae */
        double h2 = pow(r0[0]*v0[1] - r0[1]*v0[0],2) + pow(r0[0]*v0[2] - r0[2]*v0[0],2) + pow(r0[1]*v0[2] - r0[2]*v0[1],2); 
        double p = h2/mu;
        double s = atan2(1.0,(3.0*sqrt(mu/pow(p,3.0))*dt))/2.0;
        double w = atan(pow(tan(s),1.0/3.0));
        xi = sqrt(p)*2.0/tan(2.0*w);
        alpha = 0.0;
    } else{
        /*hyperbolae*/
        double a = 1.0/alpha;
        double sn = dt/fabs(dt);
        xi = sn*sqrt(-a)*log(-2*mu*alpha*dt/(nu0 + sn*sqrt(-mu*alpha)*(1.0 - r0norm*alpha)));
    }


    /*loop*/
    int counter = 0;
    double r = r0norm;
    double xiup = MAX(fabs(xi),fabs(r))*10.0;
    double psi, c2, c3, psi12;
    while ((fabs(xiup) > epsmult*EPS(MAX(fabs(xi),fabs(r)))) && (counter < 1000)){

        psi = pow(xi,2.0)*alpha;
        psi12 = sqrt(fabs(psi));
        if (psi >= 0){
            c2 =  (1 - cos(psi12))/psi;
            c3 = (psi12 - sin(psi12))/pow(psi12,3.0);
        }else{
            c2 = (1 - cosh(psi12))/psi;
            c3 = (sinh(psi12) - psi12)/pow(psi12,3.0);
        }

        if (c2+c3 == 0.){
            c2 = 1.0/2.0;
            c3 = 1.0/6.0;
        }

        r = pow(xi,2.0)*c2 + nu0osmu*xi*(1 - psi*c3) + r0norm*(1 - psi*c2);
        xiup = (sqrt(mu)*dt - pow(xi,3.0)*c3 - nu0osmu*pow(xi,2.0)*c2 - r0norm*xi*(1 - psi*c3))/r;

        xi += xiup;
        counter += 1;

    }
    if (counter == 1000){
        /*printf("Failed to converge on xi.");
        printf("xiup = %6.6e\n", xiup);*/
        return -3;
    }

    double f = 1.0 - pow(xi,2.0)/r0norm*c2;
    double g = dt - pow(xi,3.0)/sqrt(mu)*c3;
    double F = sqrt(mu)/r/r0norm*xi*(psi*c3 - 1.0);
    double G = 1.0 - pow(xi,2.0)/r*c2;
        
    int i;
    for (i=0; i<3; i++) {
        x1[i] = x0[i]*f + x0[i+3]*g;
    }
    for (i=3; i<6; i++) {
        x1[i] = x0[i-3]*F + x0[i]*G;
    }
    
    return 0;

}
