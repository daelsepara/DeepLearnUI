using DeepLearnCS;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace DeepLearnUI
{
    class Feature
    {
        public static Bitmap Get(ManagedCNN cnn, int layer, int i, int j)
        {
            if (layer >= 0 && layer < cnn.Layers.Count && i >= 0 && i < cnn.Layers[layer].FeatureMap.i && j >= 0 && j < cnn.Layers[layer].FeatureMap.j)
            {
                var FeatureMap = new ManagedArray(cnn.Layers[layer].FeatureMap.x, cnn.Layers[layer].FeatureMap.y, cnn.Layers[layer].FeatureMap.z);
                var Transposed = new ManagedArray(FeatureMap);
                var bitmap = new Bitmap(cnn.Layers[layer].FeatureMap.x, cnn.Layers[layer].FeatureMap.y, PixelFormat.Format24bppRgb);

                ManagedOps.Copy4DIJ2D(FeatureMap, cnn.Layers[layer].FeatureMap, i, j);
                ManagedMatrix.Transpose(Transposed, FeatureMap);

                // Get normalization values
                double min = 1.0;
                double max = 0.0;

                for (int y = 0; y < Transposed.y; y++)
                {
                    for (int x = 0; x < Transposed.x; x++)
                    {
                        if (Transposed[x, y] > max)
                            max = Transposed[x, y];

                        if (Transposed[x, y] < min)
                            min = Transposed[x, y];
                    }
                }

                Draw(bitmap, Transposed, min, max);

                ManagedOps.Free(FeatureMap, Transposed);

                return bitmap;
            }

            // return empty bitmap
            return new Bitmap(1, 1, PixelFormat.Format24bppRgb);
        }

        static void Draw(Bitmap bitmap, ManagedArray Activation, double min, double max)
        {
            Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            BitmapData bmpData = bitmap.LockBits(rect, ImageLockMode.ReadWrite, bitmap.PixelFormat);

            var Depth = Image.GetPixelFormatSize(bitmap.PixelFormat);
            var Channels = Depth / 8;

            for (int y = 0; y < Activation.y; y++)
            {
                for (int x = 0; x < Activation.x; x++)
                {
                    var startIndex = y * bmpData.Stride + x * Channels;

                    if (max - min != 0.0)
                    {
                        var DoubleVal = 255.0 * (Activation[x, y] - min) / (max - min);
                        var ByteVal = Convert.ToByte(DoubleVal);

                        Marshal.WriteByte(bmpData.Scan0, startIndex, ByteVal);

                        if (Depth == 32 || Depth == 24)
                        {
                            Marshal.WriteByte(bmpData.Scan0, startIndex + 1, ByteVal);
                            Marshal.WriteByte(bmpData.Scan0, startIndex + 2, ByteVal);
                        }
                    }
                }
            }

            bitmap.UnlockBits(bmpData);
        }
    }
}
