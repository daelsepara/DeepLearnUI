using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;

namespace DeepLearnCS
{
    public class ManagedLayerJSON
    {
        public int Type;
        public int OutputMaps;
        public int Scale;
        public int KernelSize;

        public double[,,,] FeatureMap; // FeatureMap[i][j][x][y]
        public double[] Bias;
    }

    public class ManagedCNNJSON
    {
        public List<ManagedLayerJSON> Layers = new List<ManagedLayerJSON>();

        public double[,] Weights;
        public double[] Bias;
    }

    public class ManagedNNJSON
    {
        public double[,] Wji;
        public double[,] Wkj;
    }

    public static class Utility
    {
        public static double[] Convert1D(ManagedArray A)
        {
            var model = new double[A.Length()];

            for (var i = 0; i < A.Length(); i++)
                model[i] = A[i];

            return model;
        }

        public static double[,] Convert2D(ManagedArray A)
        {
            var model = new double[A.y, A.x];

            for (var y = 0; y < A.y; y++)
                for (var x = 0; x < A.x; x++)
                    model[y, x] = A[x, y];

            return model;
        }

        public static double[,,] Convert3D(ManagedArray A)
        {
            var model = new double[A.y, A.x, A.z];

            for (var z = 0; z < A.z; z++)
                for (var y = 0; y < A.y; y++)
                    for (var x = 0; x < A.x; x++)
                        model[y, x, z] = A[x, y, z];

            return model;
        }

        public static double[,,,] Convert4DIJ(ManagedArray A)
        {
            var model = new double[A.i, A.j, A.y, A.x];

            var temp = new ManagedArray(A.x, A.y);

            for (var i = 0; i < A.i; i++)
            {
                for (var j = 0; j < A.j; j++)
                {
                    ManagedOps.Copy4DIJ2D(temp, A, i, j);

                    for (var y = 0; y < A.y; y++)
                        for (var x = 0; x < A.x; x++)
                            model[i, j, y, x] = temp[x, y];
                }
            }

            ManagedOps.Free(temp);

            return model;
        }

        public static ManagedArray Set(double[] A, bool vert = false)
        {
            var ii = A.GetLength(0);

            var model = vert ? new ManagedArray(1, ii) : new ManagedArray(ii);

            for (var i = 0; i < ii; i++)
                model[i] = A[i];

            return model;
        }

        public static ManagedArray Set(double[,] A)
        {
            var yy = A.GetLength(0);
            var xx = A.GetLength(1);

            var model = new ManagedArray(xx, yy);

            for (var y = 0; y < yy; y++)
                for (var x = 0; x < xx; x++)
                    model[x, y] = A[y, x];

            return model;
        }

        public static ManagedArray Set(double[,,] A)
        {
            var yy = A.GetLength(0);
            var xx = A.GetLength(1);
            var zz = A.GetLength(2);

            var model = new ManagedArray(xx, yy, zz);

            for (var z = 0; z < zz; z++)
                for (var y = 0; y < yy; y++)
                    for (var x = 0; x < xx; x++)
                        model[x, y, z] = A[y, x, z];

            return model;
        }

        public static ManagedArray Set(double[,,,] A)
        {
            var ii = A.GetLength(0);
            var jj = A.GetLength(1);
            var yy = A.GetLength(2);
            var xx = A.GetLength(3);

            var model = new ManagedArray(xx, yy, 1, ii, jj);

            var temp = new ManagedArray(xx, yy);

            for (var i = 0; i < ii; i++)
            {
                for (var j = 0; j < jj; j++)
                {
                    for (var y = 0; y < yy; y++)
                        for (var x = 0; x < xx; x++)
                            temp[x, y] = A[i, j, y, x];

                    ManagedOps.Copy2D4DIJ(model, temp, i, j);
                }
            }

            ManagedOps.Free(temp);

            return model;
        }

        public static ManagedLayerJSON Convert(ManagedLayer layer)
        {
            var model = new ManagedLayerJSON
            {
                Type = (int)layer.Type
            };

            if (layer.Type == LayerTypes.Convolution)
            {
                model.OutputMaps = layer.OutputMaps;
                model.KernelSize = layer.KernelSize;
                model.FeatureMap = Convert4DIJ(layer.FeatureMap);
                model.Bias = Convert1D(layer.Bias);
            }

            if (layer.Type == LayerTypes.Subsampling)
            {
                model.Scale = layer.Scale;
            }

            return model;
        }

        public static ManagedCNNJSON Convert(ManagedCNN network)
        {
            var model = new ManagedCNNJSON
            {
                Layers = new List<ManagedLayerJSON>(),
                Weights = Convert2D(network.Weights),
                Bias = Convert1D(network.Bias)
            };

            foreach (var layer in network.Layers)
            {
                model.Layers.Add(Convert(layer));
            }

            return model;
        }

        public static ManagedNNJSON Convert(ManagedNN network)
        {
            var model = new ManagedNNJSON()
            {
                Wji = Convert2D(network.Wji),
                Wkj = Convert2D(network.Wkj)
            };

            return model;
        }

        public static string Serialize(ManagedCNN network)
        {
            var model = Convert(network);

            string output = JsonConvert.SerializeObject(model);

            return output;
        }

        public static string Serialize(ManagedNN network)
        {
            var model = Convert(network);

            string output = JsonConvert.SerializeObject(model);

            return output;
        }

        public static ManagedCNN DeserializeCNN(string json)
        {
            var model = JsonConvert.DeserializeObject<ManagedCNNJSON>(json);

            var network = new ManagedCNN();

            foreach (var layer in model.Layers)
            {
                if (layer.Type == (int)LayerTypes.Input)
                {
                    network.Layers.Add(new ManagedLayer());
                }

                if (layer.Type == (int)LayerTypes.Convolution)
                {
                    network.Layers.Add(new ManagedLayer(layer.OutputMaps, layer.KernelSize));

                    var index = network.Layers.Count;

                    network.Layers[index - 1].FeatureMap = Set(layer.FeatureMap);
                    network.Layers[index - 1].Bias = Set(layer.Bias, false);
                }

                if (layer.Type == (int)LayerTypes.Subsampling)
                {
                    network.Layers.Add(new ManagedLayer(layer.Scale));
                }
            }

            network.Weights = Set(model.Weights);
            network.Bias = Set(model.Bias, true);

            return network;
        }

        public static ManagedNN DeserializeNN(string json)
        {
            var model = JsonConvert.DeserializeObject<ManagedNNJSON>(json);

            var network = new ManagedNN
            {
                Wji = Set(model.Wji),
                Wkj = Set(model.Wkj)
            };

            return network;
        }

        public static void Save(string BaseDirectory, string Filename, string json)
        {
            if (!string.IsNullOrEmpty(BaseDirectory) && !string.IsNullOrEmpty(Filename) && !string.IsNullOrEmpty(json))
            {
                var filename = string.Format("{0}/{1}.json", BaseDirectory, Filename);

                using (var file = new StreamWriter(filename, false))
                {
                    file.Write(json);
                }
            }
        }

        public static ManagedCNN LoadCNN(string BaseDirectory, string Filename)
        {
            var json = "";

            if (!string.IsNullOrEmpty(BaseDirectory) && !string.IsNullOrEmpty(Filename))
            {
                var filename = string.Format("{0}/{1}.json", BaseDirectory, Filename);

                using (var file = new StreamReader(filename))
                {
                    string line = "";

                    while (!string.IsNullOrEmpty(line = file.ReadLine()))
                    {
                        json += line;
                    }
                }
            }

            return !string.IsNullOrEmpty(json) ? DeserializeCNN(json) : null;
        }

        public static ManagedNN LoadNN(string BaseDirectory, string Filename)
        {
            var json = "";

            if (!string.IsNullOrEmpty(BaseDirectory) && !string.IsNullOrEmpty(Filename))
            {
                var filename = string.Format("{0}/{1}.json", BaseDirectory, Filename);

                using (var file = new StreamReader(filename))
                {
                    string line = "";

                    while (!string.IsNullOrEmpty(line = file.ReadLine()))
                    {
                        json += line;
                    }
                }
            }

            return !string.IsNullOrEmpty(json) ? DeserializeNN(json) : null;
        }
    }
}
