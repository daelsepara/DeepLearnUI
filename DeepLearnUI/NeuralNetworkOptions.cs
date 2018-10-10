namespace DeepLearnCS
{
    public class NeuralNetworkOptions
    {
        public double Alpha;
        public int Epochs;
        public int Inputs;
        public int Nodes;
        public int Items;
        public int Categories;
        public double Tolerance;

        public NeuralNetworkOptions(double alpha, int epochs, int categories, int inputs, int nodes, int items, double tolerance)
        {
            Alpha = alpha;
            Epochs = epochs;
            Inputs = inputs; // Input layer features (i)
            Nodes = nodes; // Hidden layer nodes (j)
            Items = items;  // number of input items
            Categories = categories; // number of output categories (k)
            Tolerance = tolerance;
        }

        public NeuralNetworkOptions()
        {
            Alpha = 1.0;
            Epochs = 1;
            Inputs = 2; // Input layer features (i)
            Nodes = 16; // Hidden layer nodes (j)
            Items = 50;  // number of input items
            Categories = 2; // number of output categories (k)
            Tolerance = 0.001;
        }
    }
}
