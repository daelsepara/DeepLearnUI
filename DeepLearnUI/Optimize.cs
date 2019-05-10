using System;
using System.Globalization;

public class FuncOutput
{
    public double Error;
    public double[] Gradient;

    public FuncOutput(double error, double[] X)
    {
        Gradient = X;
        Error = error;
    }
}

// Minimize a continuous differentialble multivariate function. Starting point
// is given by "X" (D by 1), and the function named in the string "f", must
// return a function value and a vector of partial derivatives. The Polack-
// Ribiere flavour of conjugate gradients is used to compute search directions,
// and a line search using quadratic and cubic polynomial approximations and the
// Wolfe-Powell stopping criteria is used together with the slope ratio method
// for guessing initial step sizes. Additionally a bunch of checks are made to
// make sure that exploration is taking place and that extrapolation will not
// be unboundedly large. The "length" gives the length of the run: if it is
// positive, it gives the maximum number of line searches, if negative its
// absolute gives the maximum allowed number of function evaluations. You can
// (optionally) give "length" a second component, which will indicate the
// reduction in function value to be expected in the first line-search (defaults
// to 1.0). The function returns when either its length is up, or if no further
// progress can be made (ie, we are at a minimum, or so close that due to
// numerical problems, we cannot get any closer). If the function terminates
// within a few iterations, it could be an indication that the function value
// and derivatives are not consistent (ie, there may be a bug in the
// implementation of your "f" function). The function returns the found
// solution "X", a vector of function values "fX" indicating the progress made
// and "i" the number of iterations (line searches or function evaluations,
// depending on the sign of "length") used.
//
// Usage: [X, fX, i] = fmincg(f, X, options)
//
// See also: checkgrad 
//
// Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
//
//
// (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
// 
// Permission is granted for anyone to copy, use, or modify these
// programs and accompanying documents for purposes of research or
// education, provided this copyright notice is retained, and note is
// made of any changes that have been made.
// 
// These programs and documents are distributed without any warranty,
// express or implied.  As the programs were written for research
// purposes only, they have not been tested to the degree that would be
// advisable in any important application.  All use of these programs is
// entirely at the user's own risk.
//
// Original C# implementation by Peter Sergio Larsen to work with Accord.NET framework
// see: https://github.com/accord-net/framework/blob/master/Sources/Extras/Accord.Math.Noncommercial/NonlinearConjugateGradient.cs
//
// Changes by [sdsepara, 2018]: 
//
// 1) Function to minimize must return a result of type FuncOutput (see above)
// 2) success and ls_failed changed to type bool, and M to type int. 
// 3) modified to work with NeuralNetworkClassifier
// 4) each call to StepOptimizer executes just one cycle of optimization
// 5) implemented Multiply, Add, Copy helper functions
//
public class Optimize
{
    // RHO and SIG are the constants in the Wolfe-Powell conditions
    readonly double RHO = 0.01;
    readonly double SIG = 0.5;

    // don't reevaluate within 0.1 of the limit of the current bracket
    readonly double INT = 0.1;

    // extrapolate maximum 3 times the current bracket
    readonly double EXT = 3.0;

    // max 20 function evaluations per line search
    readonly int MAX = 20;

    // maximum allowed slope ratio
    readonly double RATIO = 100.0;

    // reduction parameter
    readonly double Red = 1.0;

    double[] s;
    double[] df1;

    public int MaxIterations;
    public int Iterations;
    public int Evaluations;

    int length;
    int M;
    int iteration;
    bool ls_failed;

    public double f1;

    double[] X0;
    double[] DF0;

    double d1;
    double z1;

    double Multiply(double[] a, double[] b)
    {
        if (a.Length == b.Length)
        {
            var dot = 0.0;

            for (var i = 0; i < a.Length; i++)
                dot += a[i] * b[i];

            return dot;
        }

        return 0.0;
    }

    void Add(double[] dst, double[] src, double scale = 1)
    {
        if (dst.Length == src.Length)
        {
            for (var i = 0; i < dst.Length; i++)
                dst[i] += scale * src[i];
        }
    }

    void Copy(double[] dst, double[] src, double scale = 1)
    {
        if (dst.Length == src.Length)
        {
            for (var i = 0; i < dst.Length; i++)
                dst[i] = scale * src[i];
        }
    }

    public void Setup(Func<double[], FuncOutput> F, double[] X)
    {
        s = new double[X.Length];

        Evaluations = 0;
        Iterations = 0;

        length = MaxIterations;
        M = 0;
        iteration = 0; // zero the run length counter
        ls_failed = false; // no previous line search has failed

        // get function value and gradient
        var eval = F(X);
        f1 = eval.Error;
        df1 = eval.Gradient;

        Evaluations++;

        // count epochs?!
        if (length < 0)
            iteration++;

        // search direction is steepest
        Copy(s, df1, -1.0);

        // this is the slope
        d1 = -Multiply(s, s);

        // initial step is red / (|s|+1)
        z1 = Red / (1.0 - d1);

        X0 = new double[X.Length];
        DF0 = new double[X.Length];
    }

    public bool Step(Func<double[], FuncOutput> F, double[] X)
    {
        // from R/Matlab smallest non-zero normalized floating point number
        var realmin = 2.225074e-308;

        // count iterations?!
        if (length > 0)
            iteration++;

        Iterations = iteration;

        // make a copy of current values
        Copy(X0, X);
        Copy(DF0, df1);

        var F0 = f1;

        // begin line search
        Add(X, s, z1);

        // evaluate cost - and gradient function with new params
        var eval = F(X);
        
        var f2 = eval.Error;
        var df2 = eval.Gradient;

        Evaluations++;

        // count epochs?!
        if (length < 0)
            iteration++;

        // initialize point 3 equal to point 1
        var d2 = Multiply(df2, s);

        var f3 = f1;
        var d3 = d1;
        var z3 = -z1;

        if (length > 0)
        {
            M = MAX;
        }
        else
        {
            M = Math.Min(MAX, -length - iteration);
        }

        // initialize quantities
        var success = false;
        var limit = -1.0;

        while (true)
        {
            while (((f2 > f1 + z1 * RHO * d1) || (d2 > -SIG * d1)) && (M > 0))
            {
                // tighten bracket
                limit = z1;

                var A = 0.0;
                var B = 0.0;
                var z2 = 0.0;

                if (f2 > f1)
                {
                    // quadratic fit 
                    z2 = z3 - ((0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3));
                }
                else
                {
                    // cubic fit
                    A = (6.0 * (f2 - f3)) / (z3 + (3.0 * (d2 + d3)));
                    B = (3.0 * (f3 - f2) - (z3 * ((d3 + 2.0) * d2)));

                    // numerical error possible - ok!
                    z2 = Math.Sqrt(((B * B) - (A * d2 * z3)) - B) / A;
                }

                if (double.IsNaN(z2) || double.IsInfinity(z2) || double.IsNegativeInfinity(z2))
                {
                    // if we had a numerical problem then bisect
                    z2 = z3 / 2.0;
                }

                // don't accept too close to limit
                z2 = Math.Max(Math.Min(z2, INT * z3), (1.0 - INT) * z3);

                // update the step
                z1 = z1 + z2;

                Add(X, s, z2);

                eval = F(X);
                f2 = eval.Error;
                df2 = eval.Gradient;
                Evaluations++;

                M = M - 1;

                // count epochs?!
                if (length < 0)
                    iteration++;

                d2 = Multiply(df2, s);

                // z3 is now relative to the location of z2
                z3 = z3 - z2;
            }

            if (f2 > (f1 + z1 * RHO * d1) || d2 > (-SIG * d1))
            {
                // this is a failure
                break;
            }

            if (d2 > (SIG * d1))
            {
                // success
                success = true;

                break;
            }

            if (M == 0)
            {
                // failure
                break;
            }

            // make cubic extrapolation
            var A1 = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3);
            var B1 = 3.0 * (f3 - f2) - z3 * (d3 + 2.0 * d2);

            // num error possible - ok!
            var z21 = -d2 * z3 * z3 / (B1 + Math.Sqrt(B1 * B1 - A1 * d2 * z3 * z3));

            if (z21 < 0.0)
            {
                z21 = z21 * -1.0;
            }

            // num prob or wrong sign?
            if (double.IsNaN(z21) || double.IsInfinity(z21) || z21 < 0)
            {
                // if we have no upper limit
                if (limit < -0.5)
                {
                    // then extrapolate the maximum amount
                    z21 = z1 * (EXT - 1.0);
                }
                else
                {
                    // otherwise bisect
                    z21 = (limit - z1) / 2.0;
                }
            }
            else if (limit > -0.5 && (z21 + z1 > limit))
            {
                // extrapolation beyond limit?

                // set to extrapolation limit
                z21 = (limit - z1) / 2.0;
            }
            else if (limit < -0.5 && (z21 + z1 > z1 * EXT))
            {
                z21 = z1 * (EXT - 1.0);
            }
            else if (z21 < -z3 * INT)
            {
                // too close to limit?
                z21 = -z3 * INT;
            }
            else if ((limit > -0.5) && (z21 < (limit - z1) * (1 - INT)))
            {
                z21 = (limit - z1) * (1.0 - INT);
            }

            // set point 3 equal to point 2
            f3 = f2;
            d3 = d2;
            z3 = -z21;
            z1 = z1 + z21;

            // update current estimates
            Add(X, s, z21);

            // evaluate functions
            eval = F(X);
            df2 = eval.Gradient;
            f2 = eval.Error;

            M = M - 1;

            // count epochs?!
            iteration = iteration + (length < 0 ? 1 : 0);

            d2 = Multiply(df2, s);

            // end of line search
        }

        // if line searched succeeded 
        if (success) {
            f1 = f2;

            // Polack-Ribiere direction
            var part1 = Multiply(df2, df2);
            var part2 = Multiply(df1, df2);
            var part3 = Multiply(df1, df1);

            Copy(s, s, (part1 - part2) / part3);
            Add(s, df2, -1.0);

            // swap derivatives
            var tmp = df1;
            df1 = df2;
            df2 = tmp;

            // get slope
            d2 = Multiply(df1, s);

            // new slope must be negative 
            if (d2 > 0.0)
            {
                // use steepest direction
                Copy(s, df1, -1.0);

                d2 = -Multiply(s, s);
            }

            // slope ratio but max RATIO
            z1 = z1 * Math.Min(RATIO, (d1 / (d2 - realmin)));

            d1 = d2;

            // this line search did not fail
            ls_failed = false;
        }
        else
        {
            // restore point from before failed line search
            f1 = F0;

            Copy(X, X0);
            Copy(df1, DF0);

            // line search twice in a row
            if (ls_failed || iteration > Math.Abs(length))
            {
                // or we ran out of time, so we give up
                return true;
            }

            // swap derivatives
            var tmp = df1;
            df1 = df2;
            df2 = tmp;

            // try steepest
            Copy(s, df1, -1.0);

            d1 = -Multiply(s, s);

            z1 = 1.0 / (1.0 - d1);

            // this line search failed
            ls_failed = true;
        }

        return !(iteration < Math.Abs(length));
    }
}
