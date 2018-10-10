using System;
using System.Collections.Generic;

namespace DeepLearnCS
{
    public class ManagedCNN
    {
        public List<ManagedLayer> Layers = new List<ManagedLayer>();

        public ManagedArray Weights = null;
        public ManagedArray WeightsDelta = null;
        public ManagedArray WeightsTransposed = null;

        public ManagedArray FeatureVector = null;
        public ManagedArray FeatureVectorDelta = null;

        public ManagedArray Output = null;
        public ManagedArray OutputError = null;
        public ManagedArray OutputDelta = null;

        // 1D
        public ManagedArray Bias = null;
        public ManagedArray BiasDelta = null;

        // Error
        public double L = 0.0;

        public List<double> rL = new List<double>();

        // Add Layer and Initialize
        public void AddLayer(ManagedLayer layer)
        {
            if (layer.Type == LayerTypes.Input)
            {
                Layers.Add(layer);
            }

            if (layer.Type == LayerTypes.Subsampling)
            {
                if (layer.Scale > 0)
                {
                    Layers.Add(layer);
                }
            }

            if (layer.Type == LayerTypes.Convolution)
            {
                if (layer.KernelSize > 0 && layer.OutputMaps > 0)
                {
                    Layers.Add(layer);
                }
            }
        }

        public void Rand(ManagedArray rand, Random random, int fan_in, int fan_out)
        {
            for (int x = 0; x < rand.Length(); x++)
            {
                rand[x] = (random.NextDouble() - 0.5) * 2.0 * Math.Sqrt(6.0 / (fan_in + fan_out));
            }
        }

        public void Setup(ManagedArray input, int classes)
        {
            var random = new Random(Guid.NewGuid().GetHashCode());

            var InputMaps = 1;

            var MapSizeX = input.x;
            var MapSizeY = input.y;

            for (int l = 0; l < Layers.Count; l++)
            {
                if (Layers[l].Type == LayerTypes.Subsampling)
                {
                    MapSizeX = MapSizeX / Layers[l].Scale;
                    MapSizeY = MapSizeY / Layers[l].Scale;
                }

                if (Layers[l].Type == LayerTypes.Convolution)
                {
                    MapSizeX = MapSizeX - Layers[l].KernelSize + 1;
                    MapSizeY = MapSizeY - Layers[l].KernelSize + 1;

                    Layers[l].FeatureMap = new ManagedArray(Layers[l].KernelSize, Layers[l].KernelSize, 1, InputMaps, Layers[l].OutputMaps);

                    var fan_out = Layers[l].OutputMaps * Layers[l].KernelSize * Layers[l].KernelSize;

                    for (int j = 0; j < Layers[l].OutputMaps; j++)
                    {
                        var fan_in = InputMaps * Layers[l].KernelSize * Layers[l].KernelSize;

                        for (int i = 0; i < InputMaps; i++)
                        {
                            var rand = new ManagedArray(Layers[l].KernelSize, Layers[l].KernelSize);
                            Rand(rand, random, fan_in, fan_out);
                            ManagedOps.Copy2D4DIJ(Layers[l].FeatureMap, rand, i, j);
                            ManagedOps.Free(rand);
                        }
                    }

                    Layers[l].Bias = new ManagedArray(Layers[l].OutputMaps);
                    InputMaps = Layers[l].OutputMaps;
                }
            }

            // 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
            // 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
            // 'ffb' is the biases of the output neurons.
            // 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)

            var fvnum = MapSizeX * MapSizeY * InputMaps;
            var onum = classes;

            Bias = new ManagedArray(1, onum);
            Weights = new ManagedArray(fvnum, onum);
            Rand(Weights, random, fvnum, onum);
        }

        // Compute Forward Transform on 3D Input
        public void FeedForward(ManagedArray batch, bool pool = false)
        {
            var n = Layers.Count;

            var InputMaps = 1;

            ManagedOps.Free(Layers[0].Activation);
            Layers[0].Activation = new ManagedArray(batch, false);

            ManagedOps.Copy4D3D(Layers[0].Activation, batch, 0);

            for (int l = 1; l < n; l++)
            {
                if (Layers[l].Type == LayerTypes.Convolution)
                {
                    var zx = Layers[l - 1].Activation.x - Layers[l].KernelSize + 1;
                    var zy = Layers[l - 1].Activation.y - Layers[l].KernelSize + 1;
                    var zz = batch.z;

                    ManagedOps.Free(Layers[l].Activation);
                    Layers[l].Activation = new ManagedArray(zx, zy, zz, Layers[l].OutputMaps, 1, false);

                    var Activation = new ManagedArray(Layers[l - 1].Activation.x, Layers[l - 1].Activation.y, batch.z, false);
                    var FeatureMap = new ManagedArray(Layers[l].KernelSize, Layers[l].KernelSize, false);

                    // create temp output map
                    var z = new ManagedArray(zx, zy, zz);
                    var ztemp = new ManagedArray(zx, zy, zz, false);

                    // !!below can probably be handled by insane matrix operations
                    for (int j = 0; j < Layers[l].OutputMaps; j++) // for each output map
                    {
                        ManagedOps.Set(z, 0.0);

                        for (int i = 0; i < InputMaps; i++)
                        {
                            // copy Layers
                            ManagedOps.Copy4D3D(Activation, Layers[l - 1].Activation, i);
                            ManagedOps.Copy4DIJ2D(FeatureMap, Layers[l].FeatureMap, i, j);

                            // convolve with corresponding kernel and add to temp output map
                            ManagedConvolution.Valid(Activation, FeatureMap, ztemp);
                            ManagedMatrix.Add(z, ztemp);
                        }

                        // add bias, pass through nonlinearity
                        ManagedMatrix.Add(z, Layers[l].Bias[j]);
                        var sigm = ManagedMatrix.Sigm(z);
                        ManagedOps.Copy3D4D(Layers[l].Activation, sigm, j);

                        ManagedOps.Free(sigm);
                    }

                    ManagedOps.Free(Activation, FeatureMap, z, ztemp);

                    InputMaps = Layers[l].OutputMaps;
                }
                else if (Layers[l].Type == LayerTypes.Subsampling)
                {
                    // downsample

                    // generate downsampling kernel
                    var scale = (double)(Layers[l].Scale * Layers[l].Scale);
                    var FeatureMap = new ManagedArray(Layers[l].Scale, Layers[l].Scale, false);
                    ManagedOps.Set(FeatureMap, 1.0 / scale);

                    ManagedOps.Free(Layers[l].Activation);
                    Layers[l].Activation = new ManagedArray(Layers[l - 1].Activation.x / Layers[l].Scale, Layers[l - 1].Activation.y / Layers[l].Scale, batch.z, InputMaps, 1);

                    var Activation = new ManagedArray(Layers[l - 1].Activation.x, Layers[l - 1].Activation.y, batch.z, false);
                    var z = new ManagedArray(Layers[l - 1].Activation.x - Layers[l].Scale + 1, Layers[l - 1].Activation.y - Layers[l].Scale + 1, batch.z, false);

                    for (int j = 0; j < InputMaps; j++)
                    {
                        // copy Layers
                        ManagedOps.Copy4D3D(Activation, Layers[l - 1].Activation, j);

                        // Subsample
                        ManagedConvolution.Valid(Activation, FeatureMap, z);

                        if (pool)
                        {
                            ManagedOps.Pool3D4D(Layers[l].Activation, z, j, Layers[l].Scale);
                        }
                        else
                        {
                            ManagedOps.Copy3D4D(Layers[l].Activation, z, j, Layers[l].Scale);
                        }
                    }

                    ManagedOps.Free(Activation, FeatureMap, z);
                }
            }

            var MapSize = Layers[n - 1].Activation.x * Layers[n - 1].Activation.y;

            ManagedOps.Free(FeatureVector);
            FeatureVector = new ManagedArray(batch.z, MapSize * Layers[n - 1].Activation.i);

            var temp1D = new ManagedArray(Layers[n - 1].Activation.y, Layers[n - 1].Activation.x, false);
            var temp2D = new ManagedArray(Layers[n - 1].Activation.x, Layers[n - 1].Activation.y, false);

            // concatenate all end layer feature maps into vector
            for (int j = 0; j < Layers[n - 1].Activation.i; j++)
            {
                for (int ii = 0; ii < batch.z; ii++)
                {
                    // Use Row-major in flattening the feature map
                    ManagedOps.Copy4D2D(temp2D, Layers[n - 1].Activation, ii, j);
                    ManagedMatrix.Transpose(temp1D, temp2D);
                    temp1D.Reshape(1, MapSize);
                    ManagedOps.Copy2DOffset(FeatureVector, temp1D, ii, j * MapSize);
                }
            }

            var WeightsFeatureVector = new ManagedArray(FeatureVector.x, Weights.y, false);
            ManagedMatrix.Multiply(WeightsFeatureVector, Weights, FeatureVector);
            var repmat = new ManagedArray(batch.z, Bias.Length(), false);
            ManagedMatrix.Expand(Bias, batch.z, 1, repmat);
            ManagedMatrix.Add(WeightsFeatureVector, repmat);

            // feedforward into output perceptrons
            ManagedOps.Free(Output);
            Output = ManagedMatrix.Sigm(WeightsFeatureVector);

            ManagedOps.Free(WeightsFeatureVector, repmat, temp1D, temp2D);
        }

        // Update Network Weights based on computed errors
        public void BackPropagation(ManagedArray batch)
        {
            var n = Layers.Count;

            // backprop deltas
            ManagedOps.Free(OutputDelta, OutputError);

            OutputDelta = new ManagedArray(Output, false);
            OutputError = new ManagedArray(Output, false);

            for (int x = 0; x < Output.Length(); x++)
            {
                // error
                OutputError[x] = Output[x] - batch[x];

                // output delta
                OutputDelta[x] = OutputError[x] * (Output[x] * (1.0 - Output[x]));
            }

            // Loss Function
            L = 0.5 * ManagedMatrix.SquareSum(OutputError) / batch.x;

            ManagedOps.Free(WeightsTransposed, FeatureVectorDelta);

            FeatureVectorDelta = new ManagedArray(FeatureVector, false);
            WeightsTransposed = new ManagedArray(Weights, false);

            // feature vector delta
            ManagedMatrix.Transpose(WeightsTransposed, Weights);
            ManagedMatrix.Multiply(FeatureVectorDelta, WeightsTransposed, OutputDelta);

            // only conv layers has sigm function
            if (Layers[n - 1].Type == LayerTypes.Convolution)
            {
                for (int x = 0; x < FeatureVectorDelta.Length(); x++)
                {
                    FeatureVectorDelta[x] = FeatureVectorDelta[x] * FeatureVector[x] * (1.0 - FeatureVector[x]);
                }
            }

            // reshape feature vector deltas into output map style
            var MapSize = Layers[n - 1].Activation.x * Layers[n - 1].Activation.y;
            var temp1D = new ManagedArray(1, MapSize, false);
            var temp2D = new ManagedArray(Layers[n - 1].Activation.x, Layers[n - 1].Activation.y, false);

            ManagedOps.Free(Layers[n - 1].Delta);
            Layers[n - 1].Delta = new ManagedArray(Layers[n - 1].Activation, false);

            for (int j = 0; j < Layers[n - 1].Activation.i; j++)
            {
                for (int ii = 0; ii < Layers[n - 1].Activation.z; ii++)
                {
                    ManagedOps.Copy2DOffsetReverse(temp1D, FeatureVectorDelta, ii, j * MapSize);
                    temp1D.Reshape(Layers[n - 1].Activation.x, Layers[n - 1].Activation.y);
                    ManagedMatrix.Transpose(temp2D, temp1D);
                    ManagedOps.Copy2D4D(Layers[n - 1].Delta, temp2D, ii, j);
                    temp1D.Reshape(1, MapSize);
                }
            }

            ManagedOps.Free(temp1D, temp2D);

            for (int l = n - 2; l >= 0; l--)
            {
                if (Layers[l].Type == LayerTypes.Convolution)
                {
                    ManagedOps.Free(Layers[l].Delta);
                    Layers[l].Delta = new ManagedArray(Layers[l].Activation, false);

                    var xx = Layers[l + 1].Scale * Layers[l + 1].Activation.x;
                    var yy = Layers[l + 1].Scale * Layers[l + 1].Activation.y;

                    var FeatureMap = new ManagedArray(Layers[l + 1].Activation.x, Layers[l + 1].Activation.y, false);
                    var FeatureMapExpanded = new ManagedArray(xx, yy, false);
                    var Activation = new ManagedArray(xx, yy, false);
                    var Delta = new ManagedArray(xx, yy, false);

                    var Scale = (1.0 / (Layers[l + 1].Scale * Layers[l + 1].Scale));

                    for (int j = 0; j < Layers[l].Activation.i; j++)
                    {
                        for (int z = 0; z < Layers[n - 1].Activation.z; z++)
                        {
                            ManagedOps.Copy4D2D(FeatureMap, Layers[l + 1].Delta, z, j);
                            ManagedMatrix.Expand(FeatureMap, Layers[l + 1].Scale, Layers[l + 1].Scale, FeatureMapExpanded);
                            ManagedOps.Copy4D2D(Activation, Layers[l].Activation, z, j);

                            for (int x = 0; x < Delta.Length(); x++)
                            {
                                Delta[x] = Activation[x] * (1 - Activation[x]) * FeatureMapExpanded[x] * Scale;
                            }

                            ManagedOps.Copy2D4D(Layers[l].Delta, Delta, z, j);
                        }
                    }

                    ManagedOps.Free(FeatureMap, FeatureMapExpanded, Activation, Delta);
                }
                else if (Layers[l].Type == LayerTypes.Subsampling)
                {
                    ManagedOps.Free(Layers[l].Delta);
                    Layers[l].Delta = new ManagedArray(Layers[l].Activation, false);

                    var Delta = new ManagedArray(Layers[l + 1].Activation.x, Layers[l + 1].Activation.y, Layers[n - 1].Activation.z);
                    var FeatureMap = new ManagedArray(Layers[l + 1].KernelSize, Layers[l + 1].KernelSize, false);
                    var rot180 = new ManagedArray(Layers[l + 1].KernelSize, Layers[l + 1].KernelSize, false);
                    var z = new ManagedArray(Layers[l].Activation.x, Layers[l].Activation.y, Layers[n - 1].Activation.z);
                    var ztemp = new ManagedArray(Layers[l].Activation.x, Layers[l].Activation.y, Layers[n - 1].Activation.z, false);

                    for (int i = 0; i < Layers[l].Activation.i; i++)
                    {
                        ManagedOps.Set(z, 0.0);

                        for (int j = 0; j < Layers[l + 1].Activation.i; j++)
                        {
                            ManagedOps.Copy4DIJ2D(FeatureMap, Layers[l + 1].FeatureMap, i, j);
                            ManagedMatrix.Rotate180(rot180, FeatureMap);

                            ManagedOps.Copy4D3D(Delta, Layers[l + 1].Delta, j);
                            ManagedConvolution.Full(Delta, rot180, ztemp);
                            ManagedMatrix.Add(z, ztemp);
                        }

                        ManagedOps.Copy3D4D(Layers[l].Delta, z, i);
                    }

                    ManagedOps.Free(Delta, FeatureMap, rot180, z, ztemp);
                }
            }

            // calc gradients
            for (int l = 1; l < n; l++)
            {
                if (Layers[l].Type == LayerTypes.Convolution)
                {
                    ManagedOps.Free(Layers[l].DeltaFeatureMap, Layers[l].DeltaBias);

                    Layers[l].DeltaFeatureMap = new ManagedArray(Layers[l].FeatureMap, false);
                    Layers[l].DeltaBias = new ManagedArray(Layers[l].OutputMaps, false);

                    var FeatureMapDelta = new ManagedArray(Layers[l].FeatureMap.x, Layers[l].FeatureMap.y, Layers[l].FeatureMap.z, false);

                    // d[j]
                    var dtemp = new ManagedArray(Layers[l].Activation.x, Layers[l].Activation.y, Layers[n - 1].Activation.z, false);

                    // a[i] and flipped
                    var atemp = new ManagedArray(Layers[l - 1].Activation.x, Layers[l - 1].Activation.y, Layers[n - 1].Activation.z, false);
                    var ftemp = new ManagedArray(Layers[l - 1].Activation.x, Layers[l - 1].Activation.y, Layers[n - 1].Activation.z, false);

                    for (int j = 0; j < Layers[l].Activation.i; j++)
                    {
                        ManagedOps.Copy4D3D(dtemp, Layers[l].Delta, j);

                        for (int i = 0; i < Layers[l - 1].Activation.i; i++)
                        {
                            ManagedOps.Copy4D3D(atemp, Layers[l - 1].Activation, i);
                            ManagedMatrix.FlipAll(ftemp, atemp);
                            ManagedConvolution.Valid(ftemp, dtemp, FeatureMapDelta);
                            ManagedMatrix.Multiply(FeatureMapDelta, 1.0 / Layers[n - 1].Activation.z);

                            ManagedOps.Copy2D4DIJ(Layers[l].DeltaFeatureMap, FeatureMapDelta, i, j);
                        }

                        Layers[l].DeltaBias[j] = ManagedMatrix.Sum(dtemp) / Layers[n - 1].Activation.z;
                    }

                    ManagedOps.Free(FeatureMapDelta, dtemp, atemp, ftemp);
                }
            }

            var FeatureVectorTransposed = new ManagedArray(FeatureVector, false);
            ManagedMatrix.Transpose(FeatureVectorTransposed, FeatureVector);

            ManagedOps.Free(WeightsDelta, BiasDelta);

            WeightsDelta = new ManagedArray(Weights, false);
            BiasDelta = new ManagedArray(Bias, false);

            ManagedMatrix.Multiply(WeightsDelta, OutputDelta, FeatureVectorTransposed);
            ManagedMatrix.Multiply(WeightsDelta, 1.0 / Layers[n - 1].Activation.z);
            ManagedMatrix.Mean(BiasDelta, OutputDelta, 0);

            ManagedOps.Free(FeatureVectorTransposed);
        }

        void ApplyGradients(ConvolutionalNeuralNetworkOptions opts)
        {
            for (int l = 1; l < Layers.Count; l++)
            {
                if (Layers[l].Type == LayerTypes.Convolution)
                {
                    ManagedMatrix.Add(Layers[l].FeatureMap, Layers[l].DeltaFeatureMap, -opts.Alpha);
                    ManagedMatrix.Add(Layers[l].Bias, Layers[l].DeltaBias, -opts.Alpha);
                }
            }

            ManagedMatrix.Add(Weights, WeightsDelta, -opts.Alpha);
            ManagedMatrix.Add(Bias, BiasDelta, -opts.Alpha);
        }

        // Count classification errors
        public int Test(ManagedArray correct, ManagedArray classifcation)
        {
            var errors = 0;

            for (int x = 0; x < Output.x; x++)
            {
                var max = 0.0;
                var cmax = 0.0;
                var index = 0;
                var cindex = 0;

                for (int y = 0; y < Output.y; y++)
                {
                    var val = Output[x, y];

                    if (val > max)
                    {
                        max = val;
                        index = y;
                    }
                }

                // Save classification
                classifcation[x] = index;

                for (int cy = 0; cy < Output.y; cy++)
                {
                    var val = correct[x, cy];

                    if (val > cmax)
                    {
                        cmax = val;
                        cindex = cy;
                    }
                }

                if (cindex != index)
                    errors++;
            }

            return errors;
        }

        // Classify data using trained network parameters and count classification errors
        public int Classify(ManagedArray test_input, ManagedArray test_output, int classes, int items, int batchsize, ManagedArray classification, bool pool = false)
        {
            var errors = 0;

            var tempx = new ManagedArray(test_input.x, test_input.y, batchsize, false);
            var tempy = new ManagedArray(batchsize, classes, false);
            var tempclass = new ManagedArray(1, batchsize, false);

            ManagedOps.Free(classification);

            classification = new ManagedArray(1, items, false);

            for (int i = 0; i < items; i += batchsize)
            {
                // generate batch
                ManagedOps.Copy3D(tempx, test_input, 0, 0, i);
                ManagedOps.Copy2D(tempy, test_output, i, 0);

                // classify
                FeedForward(tempx, pool);

                // count classifcation errors
                errors += Test(tempy, tempclass);

                // save classification
                ManagedOps.Copy2DOffset(classification, tempclass, i, 0);
            }

            ManagedOps.Free(tempx, tempy, tempclass);

            return errors;
        }

        public void Train(ManagedArray input, ManagedArray output, ConvolutionalNeuralNetworkOptions opts)
        {
            var temp_input = new ManagedArray(input.x, input.y, opts.BatchSize, false);
            var temp_output = new ManagedArray(opts.BatchSize, output.y, false);

            for (int epoch = 0; epoch < opts.Epochs; epoch++)
            {
                var start = Profiler.now();

                var rLVal = 0.0;

                rL.Clear();

                for (int i = 0; i < opts.Items; i += opts.BatchSize)
                {
                    ManagedOps.Copy3D(temp_input, input, 0, 0, i);
                    ManagedOps.Copy2D(temp_output, output, i, 0);

                    FeedForward(temp_input, opts.Pool);
                    BackPropagation(temp_output);
                    ApplyGradients(opts);

                    if (rL.Count == 0)
                    {
                        rL.Add(L);
                    }

                    rLVal = 0.99 * rL[rL.Count - 1] + 0.01 * L;

                    rL.Add(rLVal);
                }

                Console.WriteLine("epoch {0}/{1} elapsed time is {2} ms - Error: {3}", (epoch + 1).ToString("D", ManagedMatrix.ci), opts.Epochs.ToString("D", ManagedMatrix.ci), Profiler.Elapsed(start).ToString("D", ManagedMatrix.ci), rLVal.ToString("0.000000", ManagedMatrix.ci));
            }

            ManagedOps.Free(temp_input, temp_output);
        }

        public void LoadFeatureMap(string BaseDirectory, string BaseFileName, int layer, int sizex, int sizey, int sizei, int sizej)
        {
            ManagedOps.Free(Layers[layer].FeatureMap);

            Layers[layer].FeatureMap = new ManagedArray(sizex, sizey, 1, sizei, sizej);

            for (int i = 0; i < sizei; i++)
            {
                for (int j = 0; j < sizej; j++)
                {
                    var filename = string.Format("{0}/{1}{2,0:D2}{3,0:D2}.txt", BaseDirectory, BaseFileName, i + 1, j + 1);

                    ManagedFile.Load2D4D(filename, Layers[layer].FeatureMap, i, j);
                }
            }
        }

        public void SaveFeatureMap(string BaseDirectory, string BaseFileName, int layer)
        {
            for (int i = 0; i < Layers[layer].FeatureMap.i; i++)
            {
                for (int j = 0; j < Layers[layer].FeatureMap.j; j++)
                {
                    var filename = string.Format("{0}/{1}{2,0:D2}{3,0:D2}.txt", BaseDirectory, BaseFileName, i + 1, j + 1);

                    ManagedFile.Save2D4D(filename, Layers[layer].FeatureMap, i, j);
                }
            }
        }

        public void LoadFeatureMapBias(string BaseDirectory, string BaseFileName, int layer, int sizei)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedOps.Free(Layers[layer].Bias);

            Layers[layer].Bias = new ManagedArray(sizei);

            ManagedFile.Load1D(filename, Layers[layer].Bias);
        }

        public void SaveFeatureMapBias(string BaseDirectory, string BaseFileName, int layer)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Save1D(filename, Layers[layer].Bias);
        }

        public void LoadNetworkWeights(string BaseDirectory, string BaseFileName, int sizex, int sizey)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedOps.Free(Weights);

            Weights = new ManagedArray(sizex, sizey);

            ManagedFile.Load2D(filename, Weights);
        }

        public void SaveNetworkWeights(string BaseDirectory, string BaseFileName)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Save2D(filename, Weights);
        }

        public void LoadNetworkBias(string BaseDirectory, string BaseFileName, int sizeb)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedOps.Free(Bias);

            Bias = new ManagedArray(1, sizeb);

            ManagedFile.Load1DY(filename, Bias);
        }

        public void SaveNetworkBias(string BaseDirectory, string BaseFileName)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Save1DY(filename, Bias);
        }

        public ManagedArray LoadData(string BaseDirectory, string BaseFileName, int sizex, int sizey, int sizez)
        {
            var data = new ManagedArray(sizex, sizey, sizez);

            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Load3DV2(filename, data);

            return data;
        }

        public void SaveData(string BaseDirectory, string BaseFileName, ManagedArray data)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Save3D(filename, data);
        }

        public ManagedArray LoadClassification(string BaseDirectory, string BaseFileName, int sizex, int sizey)
        {
            var classification = new ManagedArray(sizex, sizey);

            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Load2DV2(filename, classification);

            return classification;
        }

        public void SaveClassification(string BaseDirectory, string BaseFileName, ManagedArray classification)
        {
            var filename = string.Format("{0}/{1}.txt", BaseDirectory, BaseFileName);

            ManagedFile.Save2D(filename, classification);
        }

        public void Free()
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                ManagedOps.Free(Layers[i]);
            }

            ManagedOps.Free(Weights, WeightsDelta, WeightsTransposed);
            ManagedOps.Free(FeatureVector, FeatureVectorDelta);
            ManagedOps.Free(Output, OutputError, OutputDelta);
            ManagedOps.Free(Bias, BiasDelta);

            rL.Clear();
            Layers.Clear();
        }
    }
}
