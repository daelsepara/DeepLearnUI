using System;

namespace DeepLearnCS
{
    public static class ManagedConvolution
    {
        public static void Full(ManagedArray input, ManagedArray filter, ManagedArray result)
        {
            var cx = input.x + filter.x - 1;
            var cy = input.y + filter.y - 1;
            var cz = input.z + filter.z - 1;

            Convolve(input, filter, result, 0, 0, 0, cx, cy, cz);
        }

        public static void Same(ManagedArray input, ManagedArray filter, ManagedArray result)
        {
            var cx = input.x + filter.x - 1;
            var cy = input.y + filter.y - 1;
            var cz = input.z + filter.z - 1;

            var dx = (double)(filter.x - 1) / 2;
            var dy = (double)(filter.y - 1) / 2;
            var dz = (double)(filter.z - 1) / 2;

            var minx = (int)Math.Ceiling(dx);
            var miny = (int)Math.Ceiling(dy);
            var minz = (int)Math.Ceiling(dz);

            var maxx = (int)Math.Ceiling(cx - dx - 1);
            var maxy = (int)Math.Ceiling(cy - dy - 1);
            var maxz = (int)Math.Ceiling(cz - dz - 1);

            var limx = maxx - minx + 1;
            var limy = maxy - miny + 1;
            var limz = maxz - minz + 1;

            Convolve(input, filter, result, minx, miny, minz, limx, limy, limz);
        }

        public static void Valid(ManagedArray input, ManagedArray filter, ManagedArray result)
        {
            var minx = filter.x - 1;
            var miny = filter.y - 1;
            var minz = filter.z - 1;

            var limx = input.x - filter.x + 1;
            var limy = input.y - filter.y + 1;
            var limz = input.z - filter.z + 1;

            Convolve(input, filter, result, minx, miny, minz, limx, limy, limz);
        }

        public static void Convolve(ManagedArray input, ManagedArray filter, ManagedArray result, int minx, int miny, int minz, int limx, int limy, int limz)
        {
            result.Resize(limx, limy, limz, false);

            if (input.x >= filter.x & input.y >= filter.y & input.z >= filter.z)
            {
                for (int ck = minz; ck < minz + limz; ck++)
                {
                    for (int cj = miny; cj < miny + limy; cj++)
                    {
                        for (int ci = minx; ci < minx + limx; ci++)
                        {
                            result[ci - minx, cj - miny, ck - minz] = 0.0;

                            for (int kz = 0; kz < input.z; kz++)
                            {
                                var boundz = ck - kz;

                                if (boundz >= 0 & boundz < filter.z & kz < input.z & kz >= 0)
                                {
                                    for (int ky = 0; ky < input.y; ky++)
                                    {
                                        var boundy = cj - ky;

                                        if (boundy >= 0 & boundy < filter.y & ky < input.y & ky >= 0)
                                        {
                                            for (int kx = 0; kx < input.x; kx++)
                                            {
                                                var boundx = ci - kx;

                                                if (boundx >= 0 & boundx < filter.x & kx < input.x & kx >= 0)
                                                {
                                                    result[ci - minx, cj - miny, ck - minz] = result[ci - minx, cj - miny, ck - minz] + input[kx, ky, kz] * filter[boundx, boundy, boundz];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
