using System;
using System.Runtime.InteropServices;

namespace DeepLearnCS
{
    unsafe public static class MemOps
    {
        public static double* New(int size, bool initialize = true)
        {
            var temp = (double*)Marshal.AllocHGlobal(size * sizeof(double));

            if (initialize)
            {
                for (int i = 0; i < size; i++)
                    temp[i] = 0;
            }

            return temp;
        }

        public static double* New(int sizex, int sizey, bool initialize = true)
        {
            return New(sizex * sizey, initialize);
        }

        public static int* IntList(int size)
        {
            var temp = (int*)Marshal.AllocHGlobal(size * sizeof(int));

            for (int i = 0; i < size; i++)
                temp[i] = i;

            return temp;
        }

        public static void Free(double* item)
        {
            if (item != null)
            {
                Marshal.FreeHGlobal((IntPtr)item);
            }

            item = null;
        }

        public static void Free(int* item)
        {
            if (item != null)
            {
                Marshal.FreeHGlobal((IntPtr)item);
            }

            item = null;

        }
    }
}
