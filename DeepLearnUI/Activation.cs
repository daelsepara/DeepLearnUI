using DeepLearnCS;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace DeepLearnUI
{
    public static class Activation
    {
        public static Bitmap Get(ManagedCNN cnn, int layer, int map)
        {
            if (layer >= 0 && layer < cnn.Layers.Count && map >= 0 && map < cnn.Layers[layer].Activation.i)
            {
                var Activation = new ManagedArray(cnn.Layers[layer].Activation.x, cnn.Layers[layer].Activation.y, cnn.Layers[layer].Activation.z);
                var Transposed = new ManagedArray(Activation);
                var bitmap = new Bitmap(cnn.Layers[layer].Activation.x, cnn.Layers[layer].Activation.y, PixelFormat.Format24bppRgb);

                ManagedOps.Copy4D2D(Activation, cnn.Layers[layer].Activation, 0, map);
                ManagedMatrix.Transpose(Transposed, Activation);

                // Get normalization values
                double min = Double.MaxValue;
                double max = Double.MinValue;

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

                ManagedOps.Free(Activation, Transposed);

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

                    if (Math.Abs(max - min) > 0)
                    {
                        var DoubleVal = 255 * (Activation[x, y] - min) / (max - min);
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
