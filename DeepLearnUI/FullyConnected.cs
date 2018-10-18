using DeepLearnCS;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace DeepLearnUI
{
    class FullyConnected
    {
        public static Bitmap Get(ManagedArray layer, bool transpose = true)
        {
            Console.WriteLine("Layer dimensions: {0} {1}", layer.x, layer.y);

            if (transpose)
            {
                var Transposed = new ManagedArray(layer, false);
                ManagedMatrix.Transpose(Transposed, layer);

                var bitmap = new Bitmap(Transposed.x, Transposed.y, PixelFormat.Format24bppRgb);

                // Get normalization values
                double min = Double.MaxValue;
                double max = Double.MinValue;

                GetNormalization(Transposed, ref min, ref max);

                Draw(bitmap, Transposed, min, max);

                ManagedOps.Free(Transposed);

                return bitmap;
            }
            else
            {
                var bitmap = new Bitmap(layer.x, layer.y, PixelFormat.Format24bppRgb);

                // Get normalization values
                double min = Double.MaxValue;
                double max = Double.MinValue;

                GetNormalization(layer, ref min, ref max);

                Draw(bitmap, layer, min, max);

                return bitmap;
            }
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

        public static void GetNormalization(ManagedArray array, ref double min, ref double max)
        {
            // Get normalization values
            min = Double.MaxValue;
            max = Double.MinValue;

            for (int y = 0; y < array.y; y++)
            {
                for (int x = 0; x < array.x; x++)
                {
                    if (array[x, y] > max)
                        max = array[x, y];

                    if (array[x, y] < min)
                        min = array[x, y];
                }
            }
        }
    }
}
