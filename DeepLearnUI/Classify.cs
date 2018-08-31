using DeepLearnCS;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace DeepLearnUI
{
    public static class Classify
    {
        public static void Bitmap(Bitmap Digit, ManagedCNN cnn, ref int digit, ref double[] Probability)
        {
            // Bitmap Data is transposed                
            var Transposed = new ManagedArray(28, 28, 1);
            var TestDigit = new ManagedArray(28, 28, 1);

            var ScaledDigit = Resize(Digit, 28, 28, true);

            Convert(ScaledDigit, TestDigit);
            ManagedMatrix.Transpose(Transposed, TestDigit);
            cnn.FeedForward(Transposed);

            digit = 0;
            var max = 0.0;

            for (int y = 0; y < cnn.Output.y; y++)
            {
                var val = cnn.Output[0, y];

                Probability[y] = val;

                if (val > max)
                {
                    max = val;
                    digit = y;
                }
            }

            ScaledDigit.Dispose();
            ManagedOps.Free(TestDigit, Transposed);
        }

        // High Quality Bitmap Resize
        public static Bitmap Resize(Bitmap image, int width, int height, bool HighQuality = false)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;

                if (HighQuality)
                {
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                }
                else
                {
                    graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                }

                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        // Convert Bitmap to Double
        public static void Convert(Bitmap bitmap, ManagedArray digit)
        {
            if (bitmap.Width == digit.x && bitmap.Height == digit.y)
            {
                Rectangle rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
                BitmapData bmpData = bitmap.LockBits(rect, ImageLockMode.ReadWrite, bitmap.PixelFormat);

                var Depth = Image.GetPixelFormatSize(bitmap.PixelFormat);
                var Channels = Depth / 8;

                for (int y = 0; y < digit.y; y++)
                {
                    for (int x = 0; x < digit.x; x++)
                    {
                        var startIndex = y * bmpData.Stride + x * Channels;

                        var r = (double)Marshal.ReadByte(bmpData.Scan0, startIndex);
                        var g = 0.0;
                        var b = 0.0;

                        // Get start index of the specified pixel
                        if (Depth == 32 || Depth == 24) // For 32 bpp get Red, Green, Blue and Alpha
                        {
                            g = Marshal.ReadByte(bmpData.Scan0, startIndex + 1);
                            b = Marshal.ReadByte(bmpData.Scan0, startIndex + 2);
                        }
                        else if (Depth == 8) // For 8 bpp get color value (Red, Green and Blue values are the same)
                        {
                            g = r;
                            b = r;
                        }

                        digit[x, y] = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0;
                    }
                }

                bitmap.UnlockBits(bmpData);
            }
        }
    }
}
