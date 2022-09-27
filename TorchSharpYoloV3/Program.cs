namespace TorchSharpYoloV3
{
    using TorchSharp;
    using static TorchSharp.torch.nn;
    using static TorchSharp.torch.utils.data;
    internal class Program
    {
        
        static void Main(string[] args)
        {
            ModelHandler mh = new ModelHandler(80, 3, 416, 3);
            // 3 anhors per scale (w,h) and 3 scales
            float[] ANCHORS = new float[] {
                0.28f, 0.22f, 0.38f, 0.48f, 0.9f, 0.78f,
                0.07f, 0.15f, 0.15f, 0.11f, 0.14f, 0.29f,
                0.02f, 0.03f, 0.04f, 0.07f, 0.08f, 0.06f };

            string root_path = "data";
            string data_set_name = "Dataset1";
            string model_weights = "weigths.dat";
            int batch_size = 5;
            
            DataLoader dataloader = mh.data_loader(root_path, data_set_name, batch_size, false);
            
            
            var params0 = mh.model().parameters();
            var param_0_4 = params0.ToArray()[4].str();

            mh.SaveModel(Path.Join(root_path, data_set_name, model_weights));

            Console.WriteLine("training...");
            mh.train(epochs: 10, data: dataloader, anchors: ANCHORS);

            var params1 = mh.model().parameters();
            var param_1_4 = params0.ToArray()[4].str();

            mh.CreateNewloadedModel(Path.Join(root_path, data_set_name, model_weights));

            var params2 = mh.model().parameters();
            var param_2_4 = params0.ToArray()[4].str();


            Console.WriteLine(param_0_4);
            Console.WriteLine(param_1_4);
            Console.WriteLine(param_2_4);
            Console.ReadLine();



            Console.WriteLine("Hello, World!");
        }
    }
}