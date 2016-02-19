#include "fmincg.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdio.h>


void printDouble(double d){
    printf("%.17f\n", d);
}

void printMatrix(Mtx d){
    std::cout<<d<<std::endl;
}
void printIj(Mtx d, int i, int j){
    printDouble(d(i,j));
}

void printFirst(Mtx d){
    printIj(d,1,1);
}

void log(std::string str){
    std::cout<<str<<std::endl;
}

using std::min;
using std::max;
const double dblMin = 2.2251e-308;
Mtx fmincg(const std::function<Cost (Mtx)> f,
            Mtx X,
            const unsigned int maxIters)
{


  Mtx::IndexVariable i, j;

  double length = maxIters;

  double red = 1;

  //zero the run length counter
  double count = 0;

  //no previous line search has failed
  double ls_failed = 0;

  std::vector<double> fX;

  //get function value and ggradient
  Cost c = f(X);
  double f1 = c.CostValue; Mtx df1 = c.Gradient;



  //count epochs?
  count += (length < 0 ? 1 : 0);

  //search direction is steepest
  Mtx s = -df1;

  //this is the slope
  Mtx _d1 = (-transpose(s) * s);
  double d1 = _d1(1,1);


  //initial step is red(|s| + 1)
  double z1 = red / (1 - d1);

  while (count < std::abs(length)) {


    //while not finished
    count += (length > 0 ? 1 : 0); //count iterations?!
  //  log("while (count < abs(length))");
    //make a copy of current values
    Mtx X0 = X; double f0 = f1; Mtx df0 = df1;
    //begin line search
    X += z1*s;

    Cost c2 = f(X);
    double f2 = c2.CostValue;
    Mtx df2 = c2.Gradient;

    //count epochs?
    count += (length < 0 ? 1 : 0);

    Mtx _d2 = transpose(df2)*s;
    double d2 = _d2(1,1);

    //initialize point 3 equal to point 1
    double f3 = f1, d3 = d1, z3 = -z1;

    double M = 0;
    if (length > 0){
      M = MAX;
    }else{
      M = std::min(MAX, (double)(-length-count));
    }

    //initialize quantities
    double success = 0;
    double limit = -1;

    while (true){
      //Mtx A, B;

      while (( (f2 > f1 + z1 * RHO * d1)
             || (d2 > -SIG*d1))
             && M > 0)
      {

        //thighten the bracket
        limit = z1;
        double z2 =0;

        bool isReal = true;
        if (f2 > f1) {
          //quadratic fit

          z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);
        }
        else
        {
          //cubic fit
          double A = 6*(f2-f3)/z3+3*(d2+d3);
          double B = 3*(f3-f2)-z3*(d3+2*d2);

          double toSquare = B*B-A*d2*z3*z3;

          if (toSquare < 0) isReal = false;

          z2 = (sqrt(toSquare) - B)/A; //numerical error possible - ok!
        }

        //if we had a numerical problem then bisect
        if (!isReal || std::isnan(z2) || std::isinf(z2)){
          z2 = z3/2;
        }
        //don't accept too close to limits
        z2 = std::max(
                std::min(z2, INT * z3),
                (1-INT) * z3);

        //update the step
        z1 += z2;

        X += z2*s;

        c2 = f(X);
        f2 = c2.CostValue;
        df2 = c2.Gradient;


        M = M - 1;
        count += (length < 0 ? 1 : 0);

        _d2 = transpose(df2) * s;
        d2 = _d2(1,1);
        z3 -= z2; //z3 is now relative to the location of z2
      }



      if (f2 > (f1 + z1 * RHO*d1) || (d2 > -SIG*d1)) {
        break; //this is a failure
      } else if (d2 > SIG * d1){
        success = 1;//success
        break;
      } else if (M == 0){
        break; //failure
      }
      //make cubic extrapolation
      double A = 0;
      double B = 0;

      A = 6*(f2-f3)/z3+3*(d2+d3);
      B = 3*(f3-f2)-z3*(d3+2*d2);

      double toSquare = B*B-A*d2*z3*z3;

      double z2 = -d2*z3*z3/(B + sqrt(toSquare));

      if (toSquare < 0 || std::isnan(z2) || std::isinf(z2) || z2 < 0){
          if (limit < -0.5){ //if we have no upper limit
            //then extrapolate the maximum amount
            z2 = z1 * (EXT - 1);
          }else{
            //otherwise bisect
            z2 = (limit - z1)/2;
          }
      }
      else if (limit > -0.5 && z2+z1 > limit){ //extrapolation beyond max?
        z2 = (limit - z1)/2; //bisect
      } else if (limit < -0.5 && z2+z1 > z1*EXT) { //beyond limit?
        z2 = z1 * (EXT - 1); //set to extrapolation limit
      } else if (z2 < -z3 * INT) {
        z2 = -z3 * INT;
      } else if (limit > -0.5 //too close to limit?
                && (z2 < (limit-z1)*(1-INT))){
        z2 = (limit-z1) * ((double)1 - INT);
      }

      f3 = f2; d3 = d2; z3 = -z2;    // set point 3 equal to point 2
      z1 = z1 + z2; X = X + z2*s;    // update current estimates

      c2 = f(X);
      f2 = c2.CostValue;
      df2 = c2.Gradient;
      M = M-1;
      count += (length < 0 ? 1 : 0); //count epochs?

      _d2 = transpose(df2) * s;
      d2 = _d2(1,1);

    }

    if (success > 0){ //if line search succeded
       // log("success");
      f1 = f2;
      fX.push_back(f1);

      //polack-ribiere direction

      Mtx _term1 = (transpose(df2) * df2 - transpose(df1) * df2);
      double term1 = _term1(1,1);

      Mtx _transposeMult = transpose(df1) * df1;
      double transposeMult = _transposeMult(1,1);


      double term1_div_df1mult = term1/transposeMult;

      Mtx term_times_s = term1_div_df1mult * s;
      Mtx term2 = term_times_s - df2;
      s = term2;

      //swap derivatives
      Mtx tmp = df1;

      df1 = df2;
      df2 = tmp;


      _d2 = transpose(df1)*s;
      d2 = _d2(1,1);

      //new slope must be neggative
      if (d2 > 0){
        s = -df1; //otherwise use steepest direction
        _d2 = (-transpose(s))*s;
        d2 = _d2(1,1);
      }

      z1 = z1 * std::min(RATIO, d1/(d2-dblMin));



      d1 = d2;
      ls_failed = 0; //this line didnt fail
    }
    else
    {

      log("failure");
      //restore point from brefore failed line search
      X = X0;
      f1 = f0;
      df1 = df0;

      //line search failed twice in a row
      //or we ran out of time, so we give up
      if (ls_failed > 0 || count > std::abs(length)){
        break;
      }
      //swap derivatives
      Mtx tmp = df1;
      df1 = df2;
      df2 = tmp;

      //try steepest

      s = -df1;
      Mtx _d1 = -transpose(s)*s;
      d1 = _d1(1,1);
      z1 = 1/(1 - d1);

      //this line search failed
      ls_failed = 1;
    }

    //std::cout<<"FMinunc last ";
    //printf("%.17f\n", c.Gradient(c.Gradient.numRows(), c.Gradient.numCols()));
  }

  return X;
}
