using System.Diagnostics;

namespace DeepLearnCS
{
    public static class Profiler
    {
        static Stopwatch timer = new Stopwatch();

        public static void Start()
        {
            timer.Restart();
        }

        public static long now()
        {
            return timer.ElapsedMilliseconds;
        }

        public static long Elapsed(long start)
        {
            var elapsed = now() - start;

            return elapsed;
        }
    }
}
