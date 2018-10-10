using System;

namespace DeepLearnCS
{
    public class ManagedNN
    {
        public ManagedArray Wji;
        public ManagedArray Wkj;

        // intermediate results
        public ManagedArray Yk;
        public ManagedArray Z2;
        public ManagedArray A2;
        public ManagedArray DeltaWji;
        public ManagedArray DeltaWkj;
        public ManagedArray Y_output;

        // Error
        public double Cost;
        public double L2;

        public int Iterations;

        Optimize Optimizer = new Optimize();

        // Forward Propagation
        public void Forward(ManagedArray training)
        {
            // add bias column to input layer
            var InputBias = new ManagedArray(1, training.y);
            ManagedOps.Set(InputBias, 1.0);

            // x = cbind(array(1, c(nrow(training_set), 1)), training_set)
            var x = ManagedMatrix.CBind(InputBias, training);

            // compute hidden layer activation

            // z_2 = x %*% t(w_ji)
            var tWji = new ManagedArray(Wji.y, Wji.x);
            ManagedMatrix.Transpose(tWji, Wji);

            Z2 = new ManagedArray(tWji.x, x.y);
            ManagedMatrix.Multiply(Z2, x, tWji);

            // z_j = nnet_sigmoid(z_2)
            var Zj = ManagedMatrix.Sigm(Z2);

            // add bias column to hidden layer output
            var HiddenBias = new ManagedArray(1, Zj.y);
            ManagedOps.Set(HiddenBias, 1.0);

            // a_2 = cbind(array(1, c(nrow(z_j), 1)), z_j)
            A2 = ManagedMatrix.CBind(HiddenBias, Zj);

            // compute output layer

            var tWkj = new ManagedArray(Wkj.y, Wkj.x);
            ManagedMatrix.Transpose(tWkj, Wkj);

            //  y_k = nnet_sigmoid(a_2 %*% t(w_kj))
            var A2Wkj = new ManagedArray(tWkj.x, A2.y);
            ManagedMatrix.Multiply(A2Wkj, A2, tWkj);

            Yk = ManagedMatrix.Sigm(A2Wkj);

            // cleanup
            ManagedOps.Free(A2Wkj, HiddenBias, InputBias);
            ManagedOps.Free(tWkj, tWji, x, Zj);
        }

        // Backward propagation
        public void BackPropagation(ManagedArray training)
        {
            // add bias column to input layer
            var InputBias = new ManagedArray(1, training.y);
            ManagedOps.Set(InputBias, 1.0);

            // x = cbind(array(1, c(nrow(training_set), 1)), training_set)
            var x = ManagedMatrix.CBind(InputBias, training);

            // compute intermediate delta values per layer

            // d3 = y_k - y_matrix
            var D3 = ManagedMatrix.Diff(Yk, Y_output);

            //  d2 = d3 %*% w_kj[, 2:ncol(w_kj)] * nnet_dsigmoid(z_2)
            var sWkj = new ManagedArray(Wkj.x - 1, Wkj.y);
            ManagedOps.Copy2DOffsetReverse(sWkj, Wkj, 1, 0);

            var D2 = new ManagedArray(sWkj.x, D3.y);
            ManagedMatrix.Multiply(D2, D3, sWkj);

            var DZ2 = ManagedMatrix.DSigm(Z2);
            ManagedMatrix.Product(D2, DZ2);

            // dWji = (t(d2) %*% x)
            // dWkj = (t(d3) % *% a_2)
            var tD2 = new ManagedArray(D2.y, D2.x);
            var tD3 = new ManagedArray(D3.y, D3.x);
            ManagedMatrix.Transpose(tD2, D2);
            ManagedMatrix.Transpose(tD3, D3);

            DeltaWji = new ManagedArray(Wji.x, Wji.y);
            DeltaWkj = new ManagedArray(Wkj.x, Wkj.y);

            ManagedMatrix.Multiply(DeltaWji, tD2, x);
            ManagedMatrix.Multiply(DeltaWkj, tD3, A2);

            // cost = sum(-y_matrix * log(y_k) - (1 - y_matrix) * log(1 - y_k))
            Cost = 0.0;
            L2 = 0.0;

            for (int i = 0; i < Y_output.Length(); i++)
            {
                L2 += Math.Sqrt(D3[i] * D3[i]);
                Cost += (-Y_output[i] * Math.Log(Yk[i]) - (1.0 - Y_output[i]) * Math.Log(1.0 - Yk[i]));
            }

            // cost = cost / m
            // dWji = dWji / m
            // dWkj = dWkj / m
            Cost /= training.y;
            L2 /= training.y;

            ManagedMatrix.Multiply(DeltaWji, 1.0 / training.y);
            ManagedMatrix.Multiply(DeltaWkj, 1.0 / training.y);

            // cleanup
            ManagedOps.Free(D2, D3, DZ2, InputBias);
            ManagedOps.Free(sWkj, tD2, tD3, x);

            // cleanup of arrays allocated in Forward
            ManagedOps.Free(A2, Yk, Z2);
        }

        public void ApplyGradients(NeuralNetworkOptions opts)
        {
            // dWji = learning_rate * dWji
            // dWkj = learning_rate * dWkj
            // w_ji = w_ji - dWji
            // w_kj = w_kj - dWkj
            ManagedMatrix.Add(Wkj, DeltaWkj, -opts.Alpha);
            ManagedMatrix.Add(Wji, DeltaWji, -opts.Alpha);

            // cleanup of arrays allocated in BackPropagation
            ManagedOps.Free(DeltaWji, DeltaWkj);
        }

        public void Rand(ManagedArray rand, Random random)
        {
            for (int x = 0; x < rand.Length(); x++)
            {
                rand[x] = (random.NextDouble() - 0.5) * 2.0;
            }
        }

        ManagedArray Labels(ManagedArray output, NeuralNetworkOptions opts)
        {
            var result = new ManagedArray(opts.Categories, opts.Items);
            var eye_matrix = ManagedMatrix.Diag(opts.Categories);

            for (int y = 0; y < opts.Items; y++)
            {
                if (opts.Categories > 1)
                {
                    for (int x = 0; x < opts.Categories; x++)
                    {
                        result[x, y] = eye_matrix[x, (int)output[y] - 1];
                    }
                }
                else
                {
                    result[y] = output[y];
                }
            }

            ManagedOps.Free(eye_matrix);

            return result;
        }

        public ManagedIntList Classify(ManagedArray test, NeuralNetworkOptions opts, double threshold = 0.5)
        {
            Forward(test);

            var classification = new ManagedIntList(test.y);

            for (int y = 0; y < test.y; y++)
            {
                if (opts.Categories > 1)
                {
                    double maxval = 0.0;
                    int maxind = 0;

                    for (int x = 0; x < opts.Categories; x++)
                    {
                        double val = Yk[x, y];

                        if (val > maxval)
                        {
                            maxval = val;
                            maxind = x;
                        }
                    }

                    classification[y] = maxind + 1;
                }
                else
                {
                    classification[y] = Yk[y] > threshold ? 1 : 0;
                }
            }

            // cleanup of arrays allocated in Forward
            ManagedOps.Free(A2, Yk, Z2);

            return classification;
        }

        public void Setup(ManagedArray output, NeuralNetworkOptions opts)
        {
            Wji = new ManagedArray(opts.Inputs + 1, opts.Nodes);
            Wkj = new ManagedArray(opts.Nodes + 1, opts.Categories);

            Y_output = Labels(output, opts);

            var random = new Random(Guid.NewGuid().GetHashCode());

            Rand(Wji, random);
            Rand(Wkj, random);

            Cost = 1.0;
            L2 = 1.0;

            Iterations = 0;
        }

        public bool Step(ManagedArray input, NeuralNetworkOptions opts)
        {
            Forward(input);
            BackPropagation(input);
            ApplyGradients(opts);

            Iterations = Iterations + 1;

            return (double.IsNaN(Cost) || Iterations >= opts.Epochs || Cost < opts.Tolerance);
        }

        public void Train(ManagedArray input, ManagedArray output, NeuralNetworkOptions opts)
        {
            Setup(output, opts);

            while (!Step(input, opts)) { }
        }

        // Reshape Network Weights for use in optimizer
        public double[] ReshapeWeights(ManagedArray A, ManagedArray B)
        {
            var X = new double[A.x * A.y + B.x * B.y];

            if (A != null && B != null)
            {
                var index = 0;

                for (var x = 0; x < A.x; x++)
                {
                    for (var y = 0; y < A.y; y++)
                    {
                        X[index] = A[x, y];

                        index++;
                    }
                }

                for (var x = 0; x < B.x; x++)
                {
                    for (var y = 0; y < B.y; y++)
                    {
                        X[index] = B[x, y];

                        index++;
                    }
                }
            }

            return X;
        }

        // Transform vector back into Network Weights
        public void ReshapeWeights(double[] X, ManagedArray A, ManagedArray B)
        {
            if (X.Length != (A.x * A.y + B.x * B.y))
                return;

            var index = 0;

            for (var x = 0; x < A.x; x++)
            {
                for (var y = 0; y < A.y; y++)
                {
                    if (index < X.Length)
                        A[x, y] = X[index];

                    index++;
                }
            }

            for (var x = 0; x < B.x; x++)
            {
                for (var y = 0; y < B.y; y++)
                {
                    if (index < X.Length)
                        B[x, y] = X[index];

                    index++;
                }
            }
        }

        ManagedArray OptimizerInput;

        public FuncOutput OptimizerCost(double[] X)
        {
            ReshapeWeights(X, Wji, Wkj);

            if (OptimizerInput != null)
                Forward(OptimizerInput);

            if (OptimizerInput != null)
                BackPropagation(OptimizerInput);

            X = ReshapeWeights(DeltaWji, DeltaWkj);

            // cleanup of arrays allocated in BackPropagation
            ManagedOps.Free(DeltaWji, DeltaWkj);

            return new FuncOutput(Cost, X);
        }

        public void SetupOptimizer(ManagedArray input, ManagedArray output, NeuralNetworkOptions opts)
        {
            Setup(output, opts);

            Optimizer.MaxIterations = opts.Epochs;

            var X = ReshapeWeights(Wji, Wkj);

            OptimizerInput = input;

            Optimizer.Setup(OptimizerCost, X);
        }

        public bool StepOptimizer(ManagedArray input, NeuralNetworkOptions opts)
        {
            OptimizerInput = input;

            var X = ReshapeWeights(Wji, Wkj);

            Optimizer.Step(OptimizerCost, X);

            Iterations = Optimizer.Iterations;

            Cost = Optimizer.f1;

            OptimizerInput = null;

            return (double.IsNaN(Cost) || Iterations >= opts.Epochs || Cost < opts.Tolerance);
        }

        public void Optimize(ManagedArray input, ManagedArray output, NeuralNetworkOptions opts)
        {
            SetupOptimizer(input, output, opts);

            while (!StepOptimizer(input, opts)) { }
        }

        public void LoadInputLayerWeights(string BaseDirectory, string BaseFileName, int sizex, int sizey)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedOps.Free(Wji);

            Wji = new ManagedArray(sizex, sizey);

            ManagedFile.Load2DV2(filename, Wji);
        }

        public void SaveInputLayerWeights(string BaseDirectory, string BaseFileName)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Save2D(filename, Wji);
        }

        public void LoadHiddenLayerWeights(string BaseDirectory, string BaseFileName, int sizex, int sizey)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedOps.Free(Wkj);

            Wkj = new ManagedArray(sizex, sizey);

            ManagedFile.Load2DV2(filename, Wkj);
        }

        public void SaveHiddenLayerWeights(string BaseDirectory, string BaseFileName)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Save2D(filename, Wkj);
        }

        public void Free()
        {
            ManagedOps.Free(Yk, Z2, A2);
            ManagedOps.Free(Wji, Wkj);
            ManagedOps.Free(Y_output);
        }
    }
}
